import json
import time
from typing import Dict, List, Any, Optional
import networkx as nx


class CapabilityKnowledgeGraph:
    """行业算法能力知识图谱：负责能力存储、子图检索(GraphRAG)、失败经验沉淀与版本演化"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._init_seed_knowledge()

    def _init_seed_knowledge(self):
        """初始化种子行业知识图谱（电信客户流失预测）"""
        self.add_node("SC_CHURN", "Scenario", {
            "name": "电信客户流失预测",
            "task_type": "binary_classification",
            "input_schema": {"numeric_cols": "List[str]", "categorical_cols": "List[str]", "label": "int (0/1)"},
            "output_schema": {"predictions": "np.ndarray", "probabilities": "np.ndarray"}
        })

        # 评价指标与业务约束节点
        self.add_node("METRIC_AUC", "Metric", {"name": "roc_auc", "threshold": 0.78, "direction": "maximize"})
        self.add_node("METRIC_F1", "Metric", {"name": "f1", "threshold": 0.62, "direction": "maximize"})
        self.add_node("CONST_LEAKAGE", "Constraint", {
            "rule": "所有缺失值填充与类别编码必须在训练集 fit，在测试集 transform，严禁全量数据提前 fit。"
        })
        self.add_node("CONST_UNSEEN", "Constraint", {
            "rule": "类别特征编码必须处理测试集中可能出现的未知类别（handle_unknown='ignore'）。"
        })

        self.add_edge("SC_CHURN", "METRIC_AUC", "REQUIRES_METRIC")
        self.add_edge("SC_CHURN", "METRIC_F1", "REQUIRES_METRIC")
        self.add_edge("SC_CHURN", "CONST_LEAKAGE", "HAS_CONSTRAINT")
        self.add_edge("SC_CHURN", "CONST_UNSEEN", "HAS_CONSTRAINT")

        # 初始算法能力节点
        capabilities = [
            ("CAP_PREP_ROBUST", "RobustImputerEncoder", "preprocessing",
             "存在缺失值与类别特征的表格数据",
             "SimpleImputer(strategy='median') + OneHotEncoder(handle_unknown='ignore')", 0.82),
            ("CAP_FE_INTERACT", "DomainRatioFeatures", "feature_engineering",
             "包含消费金额、在网时长等连续数值特征",
             "构造基础消费频次比特征", 0.83),
            ("CAP_MODEL_RF", "BalancedRandomForest", "model",
             "正负样本比例失衡的中小规模表格数据",
             "RandomForestClassifier(n_estimators=150, class_weight='balanced', max_depth=8, random_state=42)", 0.84),
            ("CAP_MODEL_GBDT", "HistGradientBoosting", "model",
             "高维非线性表格特征，需要高AUC表现与快速推理",
             "HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, class_weight='balanced', random_state=42)", 0.88),
            ("CAP_POST_THRESH", "F1OptimalThresholdTuner", "post_processing",
             "类别不平衡场景下默认 0.5 阈值导致 F1 偏低",
             "基于 PR 曲线搜索最大化 F1-score 的动态分类阈值", 0.86),
        ]
        for cid, name, cat, cond, impl, score in capabilities:
            self.add_node(cid, "Capability", {
                "name": name,
                "category": cat,
                "applicable_conditions": cond,
                "implementation_hint": impl,
                "dependencies": ["scikit-learn", "numpy", "pandas"],
                "version": "1.0.0",
                "historical_score": score
            })
            self.add_edge(cid, "SC_CHURN", "APPLICABLE_TO")

        # 组合协同边（供 Beam Search 规划器打分）
        self.add_edge("CAP_PREP_ROBUST", "CAP_FE_INTERACT", "COMPOSED_WITH", {"synergy": 0.90})
        self.add_edge("CAP_FE_INTERACT", "CAP_MODEL_GBDT", "COMPOSED_WITH", {"synergy": 0.92})
        self.add_edge("CAP_FE_INTERACT", "CAP_MODEL_RF", "COMPOSED_WITH", {"synergy": 0.85})
        self.add_edge("CAP_MODEL_GBDT", "CAP_POST_THRESH", "COMPOSED_WITH", {"synergy": 0.95})
        self.add_edge("CAP_MODEL_RF", "CAP_POST_THRESH", "COMPOSED_WITH", {"synergy": 0.88})

    def add_node(self, node_id: str, label: str, attrs: Dict[str, Any]):
        self.graph.add_node(node_id, label=label, **attrs)

    def add_edge(self, src: str, dst: str, rel: str, attrs: Optional[Dict[str, Any]] = None):
        self.graph.add_edge(src, dst, relation=rel, **(attrs or {}))

    def retrieve_context(self, query: str, scenario_id: str = "SC_CHURN") -> Dict[str, Any]:
        """GraphRAG 子图检索：检索相关能力、指标、约束与历史失败案例"""
        capabilities, metrics, constraints, failures = [], [], [], []
        for nid, data in self.graph.nodes(data=True):
            label = data.get("label")
            if label == "Capability":
                capabilities.append({"id": nid, **data})
            elif label == "Metric":
                metrics.append({"id": nid, **data})
            elif label == "Constraint":
                constraints.append({"id": nid, **data})
            elif label == "FailureCase":
                failures.append({"id": nid, **data})

        return {
            "scenario": self.graph.nodes[scenario_id],
            "capabilities": sorted(capabilities, key=lambda x: x.get("historical_score", 0), reverse=True),
            "metrics": metrics,
            "constraints": constraints,
            "failures": failures
        }

    def sink_verification_result(
        self,
        plan_ids: List[str],
        metrics_res: Dict[str, float],
        passed: bool,
        failure_info: Optional[Dict[str, str]] = None
    ) -> str:
        """能力沉淀闭环：将验证记录、新版本号及失败经验写回知识图谱"""
        rec_id = f"VER_{int(time.time() * 1000) % 100000}"
        self.add_node(rec_id, "VerificationRecord", {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "plan": plan_ids,
            "metrics": metrics_res,
            "passed": passed
        })

        for cid in plan_ids:
            if cid in self.graph.nodes:
                self.add_edge(cid, rec_id, "VERIFIED_BY")
                if passed and "roc_auc" in metrics_res:
                    old_score = self.graph.nodes[cid].get("historical_score", 0.8)
                    self.graph.nodes[cid]["historical_score"] = round(0.7 * old_score + 0.3 * metrics_res["roc_auc"], 4)
                    ver_parts = self.graph.nodes[cid].get("version", "1.0.0").split(".")
                    ver_parts[-1] = str(int(ver_parts[-1]) + 1)
                    self.graph.nodes[cid]["version"] = ".".join(ver_parts)

        if failure_info:
            fail_id = f"FAIL_{int(time.time() * 1000) % 100000}"
            self.add_node(fail_id, "FailureCase", failure_info)
            for cid in plan_ids:
                if cid in self.graph.nodes:
                    self.add_edge(cid, fail_id, "HAS_FAILURE_EXPERIENCE")

        return rec_id

    def export_json(self, path: str):
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)