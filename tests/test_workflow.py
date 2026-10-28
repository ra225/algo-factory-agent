from kg.graph_store import CapabilityKnowledgeGraph
from sandbox.evaluator import ASTSecurityAndContractChecker, VerificationSandbox, generate_synthetic_churn_dataset
from agents.workflow import MultiAgentAlgoFactory


def test_ast_security_blocker():
    """测试 AST 安全检查器能否拦截危险系统调用"""
    malicious_code = (
        "import os\n"
        "class GeneratedCapability:\n"
        "    def __init__(self): pass\n"
        "    def fit(self, X, y): os.system('echo hacked')\n"
        "    def predict(self, X): pass\n"
        "    def predict_proba(self, X): pass\n"
    )
    ok, msg = ASTSecurityAndContractChecker().check(malicious_code)
    assert not ok
    assert "安全违规" in msg


def test_end_to_end_self_healing_and_sink():
    """测试从抽取、规划、首次报错、第二轮自修复到图谱沉淀的完整闭环"""
    kg = CapabilityKnowledgeGraph()
    df = generate_synthetic_churn_dataset(n_samples=600)
    sandbox = VerificationSandbox(df=df, metric_thresholds={"roc_auc": 0.75, "f1": 0.55})
    factory = MultiAgentAlgoFactory(kg=kg, sandbox=sandbox)

    # 抽取新算子
    new_cap = factory.extract_knowledge_from_doc("data/raw_docs/churn_expert_guide.md")
    assert new_cap["id"] in kg.graph.nodes

    # 规划方案
    plans = factory.plan_candidates_beam_search("客户流失预测", beam_width=2)
    assert len(plans) == 2

    # 第 1 轮触发未知类别运行时异常
    code_v1 = factory.generate_code(plans[0], attempt=1)
    res_v1 = sandbox.evaluate_code(code_v1)
    assert not res_v1["passed"]
    assert res_v1["stage"] == "RUNTIME_EXECUTION", f"第1轮预期在运行时报错，实际为: {res_v1}"

    # 第 2 轮自动修复并通过验证
    code_v2 = factory.generate_code(plans[0], attempt=2, last_error=res_v1["error"])
    res_v2 = sandbox.evaluate_code(code_v2)
    assert res_v2["passed"], f"第2轮验证失败: {res_v2}"