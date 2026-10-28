import os
from typing import Dict, Any, List
from kg.graph_store import CapabilityKnowledgeGraph
from sandbox.evaluator import VerificationSandbox


class MultiAgentAlgoFactory:
    """多智能体算法能力工厂编排器：包含知识抽取、Beam Search 方案规划、代码复刻与自修复"""

    def __init__(self, kg: CapabilityKnowledgeGraph, sandbox: VerificationSandbox):
        self.kg = kg
        self.sandbox = sandbox
        self.api_key = os.getenv("OPENAI_API_KEY")

    def extract_knowledge_from_doc(self, doc_path: str) -> Dict[str, Any]:
        """ExtractorAgent：从业务专家文档中抽取新算子并写入知识图谱"""
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                _ = f.read()

        extracted_cap = {
            "id": "CAP_FE_CHURN_VELOCITY",
            "name": "ChurnVelocityAndRiskScore",
            "category": "feature_engineering",
            "applicable_conditions": "客户流失预测场景，包含月费、总消费、在网时长与投诉次数",
            "implementation_hint": "计算月均消费偏差比 (total_charges / (tenure * monthly_charges + 1)) 与单位时长投诉密度",
            "dependencies": ["numpy", "pandas"],
            "version": "1.0.0",
            "historical_score": 0.89
        }
        self.kg.add_node(extracted_cap["id"], "Capability", extracted_cap)
        self.kg.add_edge(extracted_cap["id"], "SC_CHURN", "APPLICABLE_TO")
        self.kg.add_edge("CAP_PREP_ROBUST", extracted_cap["id"], "COMPOSED_WITH", {"synergy": 0.94})
        self.kg.add_edge(extracted_cap["id"], "CAP_MODEL_GBDT", "COMPOSED_WITH", {"synergy": 0.93})
        self.kg.add_edge(extracted_cap["id"], "CAP_MODEL_RF", "COMPOSED_WITH", {"synergy": 0.87})
        return extracted_cap

    def plan_candidates_beam_search(self, user_requirement: str, beam_width: int = 2) -> List[Dict[str, Any]]:
        """PlannerAgent：通过图上 Beam Search 规划 Top-K 候选算法方案"""
        ctx = self.kg.retrieve_context(user_requirement, scenario_id="SC_CHURN")
        caps_by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for c in ctx["capabilities"]:
            caps_by_cat.setdefault(c["category"], []).append(c)

        stages = ["preprocessing", "feature_engineering", "model", "post_processing"]
        beams = [{"nodes": [], "score": 0.0, "rationale": []}]

        for stage in stages:
            candidates = caps_by_cat.get(stage, [])
            new_beams = []
            for beam in beams:
                for cand in candidates:
                    synergy_bonus = 0.0
                    if beam["nodes"]:
                        prev_id = beam["nodes"][-1]["id"]
                        if self.kg.graph.has_edge(prev_id, cand["id"]):
                            edge_data = self.kg.graph.get_edge_data(prev_id, cand["id"])
                            first_edge = list(edge_data.values())[0]
                            synergy_bonus = first_edge.get("synergy", 0.0) * 0.2
                    step_score = cand.get("historical_score", 0.8) + synergy_bonus
                    new_beams.append({
                        "nodes": beam["nodes"] + [cand],
                        "score": round(beam["score"] + step_score, 4),
                        "rationale": beam["rationale"] + [f"[{stage}] 选用 {cand['name']} ({cand['implementation_hint']})"]
                    })
            new_beams.sort(key=lambda x: x["score"], reverse=True)
            beams = new_beams[:beam_width]

        return beams

    def generate_code(self, plan: Dict[str, Any], attempt: int = 1, last_error: str = "") -> str:
        """CoderAgent：生成算法代码（Attempt 1 演示未见类别报错，Attempt 2 根据沙箱反馈自动修复）"""
        model_node = [n for n in plan["nodes"] if n["category"] == "model"][0]
        use_gbdt = "GBDT" in model_node["id"]

        handle_unknown_str = "'error'" if (attempt == 1 and not last_error) else "'ignore'"
        estimator_code = (
            "HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, class_weight='balanced', random_state=42)"
            if use_gbdt else
            "RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42)"
        )

        return f'''import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score


class GeneratedCapability:
    """
    自动复刻算法能力组件
    选定方案节点: {[n["id"] for n in plan["nodes"]]}
    修复轮次: Attempt #{attempt}
    """

    def __init__(self):
        self.num_cols = ["tenure", "monthly_charges", "total_charges", "support_calls",
                         "charge_velocity", "support_density"]
        self.cat_cols = ["contract_type", "payment_method"]
        self.preprocessor = None
        self.model = None
        self.best_threshold = 0.5

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        tenure_safe = X["tenure"].fillna(1.0).clip(lower=1.0)
        monthly_safe = X["monthly_charges"].fillna(50.0)
        total_safe = X["total_charges"].fillna(tenure_safe * monthly_safe)
        X["charge_velocity"] = total_safe / (tenure_safe * monthly_safe + 1.0)
        X["support_density"] = X["support_calls"].fillna(0.0) / tenure_safe
        return X

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_feat = self._engineer_features(X)
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown={handle_unknown_str}, sparse_output=False))
        ])
        self.preprocessor = ColumnTransformer([
            ("num", num_pipe, self.num_cols),
            ("cat", cat_pipe, self.cat_cols)
        ])
        X_trans = self.preprocessor.fit_transform(X_feat)
        self.model = {estimator_code}
        self.model.fit(X_trans, y)

        train_probs = self.model.predict_proba(X_trans)[:, 1]
        best_f1, best_th = 0.0, 0.5
        for th in np.linspace(0.25, 0.75, 21):
            score = f1_score(y, (train_probs >= th).astype(int), zero_division=0)
            if score > best_f1:
                best_f1, best_th = score, float(th)
        self.best_threshold = best_th
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_feat = self._engineer_features(X)
        X_trans = self.preprocessor.transform(X_feat)
        return self.model.predict_proba(X_trans)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.best_threshold).astype(int)
'''

    def reflect_on_failure(self, error_msg: str) -> Dict[str, str]:
        """ReflectorAgent：分析失败原因并提炼可复用的知识图谱经验节点"""
        return {
            "error_type": "UnseenCategoryValueError",
            "root_cause": error_msg,
            "mitigation_rule": "线上/测试集极易出现未见过的类别值，OneHotEncoder 必须显式指定 handle_unknown='ignore'。"
        }