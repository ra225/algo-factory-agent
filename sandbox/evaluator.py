import ast
import time
import tracemalloc
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


BANNED_MODULES = {"os", "subprocess", "sys", "shutil", "socket", "requests", "urllib"}
REQUIRED_METHODS = {"__init__", "fit", "predict_proba", "predict"}


class ASTSecurityAndContractChecker(ast.NodeVisitor):
    """静态检查生成代码的安全性（禁止危险系统调用）与接口规范性"""

    def __init__(self):
        self.errors = []
        self.found_class = False
        self.found_methods = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root_pkg = alias.name.split(".")[0]
            if root_pkg in BANNED_MODULES:
                self.errors.append(f"安全违规：禁止导入系统或网络模块 '{root_pkg}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root_pkg = node.module.split(".")[0]
            if root_pkg in BANNED_MODULES:
                self.errors.append(f"安全违规：禁止从 '{root_pkg}' 导入组件")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.name == "GeneratedCapability":
            self.found_class = True
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    self.found_methods.add(item.name)
        self.generic_visit(node)

    def check(self, code_str: str) -> Tuple[bool, str]:
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return False, f"语法错误 (SyntaxError): {e}"

        self.visit(tree)
        if not self.found_class:
            self.errors.append("接口违规：未找到标准实现类 'GeneratedCapability'")
        missing = REQUIRED_METHODS - self.found_methods
        if missing:
            self.errors.append(f"接口违规：'GeneratedCapability' 缺少必需方法: {missing}")

        if self.errors:
            return False, " | ".join(self.errors)
        return True, "AST 安全检查与接口契约校验通过"


def generate_synthetic_churn_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """生成具有缺失值、类别不平衡与非线性交叉关系的模拟客户流失数据"""
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 72, size=n_samples).astype(float)
    monthly_charges = rng.uniform(20.0, 120.0, size=n_samples)
    total_charges = tenure * monthly_charges + rng.normal(0, 50, size=n_samples)
    support_calls = rng.poisson(1.8, size=n_samples).astype(float)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples)

    logit = (
        -2.2
        + 0.025 * (monthly_charges - 60)
        - 0.035 * tenure
        + 0.45 * support_calls
        + np.where(contract_type == "Month-to-month", 1.25, -0.6)
        + np.where(payment_method == "Electronic check", 0.5, 0.0)
        + 0.015 * (monthly_charges / (tenure + 1.0)) * 10
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, size=n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_calls": support_calls,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "label": churn
    })

    mask = rng.uniform(0, 1, size=n_samples) < 0.05
    df.loc[mask, "total_charges"] = np.nan
    return df


class VerificationSandbox:
    """统一验证沙箱：执行功能正确性、指标表现、未见类别鲁棒性、运行耗时与内存峰值测试"""

    def __init__(self, df: pd.DataFrame, metric_thresholds: Dict[str, float]):
        self.df = df
        self.thresholds = metric_thresholds

    def evaluate_code(self, code_str: str) -> Dict[str, Any]:
        checker = ASTSecurityAndContractChecker()
        ast_ok, ast_msg = checker.check(code_str)
        if not ast_ok:
            return {"passed": False, "stage": "AST_CHECK", "error": ast_msg, "metrics": {}}

        namespace: Dict[str, Any] = {}
        try:
            exec(code_str, namespace)
            model_cls = namespace["GeneratedCapability"]
        except Exception as e:
            return {"passed": False, "stage": "CODE_LOAD", "error": f"代码加载异常: {e}", "metrics": {}}

        X = self.df.drop(columns=["label"]).copy()
        y = self.df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        X_test_stress = X_test.copy()
        X_test_stress.iloc[0, X_test_stress.columns.get_loc("payment_method")] = "Crypto_New_Unseen"

        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            instance = model_cls()
            instance.fit(X_train, y_train)
            probs = instance.predict_proba(X_test_stress)
            preds = instance.predict(X_test_stress)
        except Exception as e:
            tracemalloc.stop()
            return {"passed": False, "stage": "RUNTIME_EXECUTION", "error": f"{type(e).__name__}: {e}", "metrics": {}}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if len(probs) != len(y_test) or np.isnan(probs).any():
            return {"passed": False, "stage": "OUTPUT_CONTRACT", "error": "预测输出长度不匹配或包含 NaN", "metrics": {}}

        metrics = {
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "runtime_ms": round(elapsed_ms, 2),
            "peak_memory_mb": round(peak_mem_bytes / (1024 * 1024), 3)
        }

        failed_metrics = []
        for m_name, thresh in self.thresholds.items():
            if m_name in metrics and metrics[m_name] < thresh:
                failed_metrics.append(f"{m_name}={metrics[m_name]} (低于阈值 {thresh})")

        if failed_metrics:
            return {
                "passed": False,
                "stage": "METRIC_THRESHOLD",
                "error": "指标未达标: " + ", ".join(failed_metrics),
                "metrics": metrics
            }

        return {"passed": True, "stage": "ALL_PASSED", "error": None, "metrics": metrics}