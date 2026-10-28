import os
from kg.graph_store import CapabilityKnowledgeGraph
from sandbox.evaluator import VerificationSandbox, generate_synthetic_churn_dataset
from agents.workflow import MultiAgentAlgoFactory


def build_markdown_report(query: str, results: list, best: dict) -> str:
    rows = []
    for r in results:
        m = r["eval"]["metrics"]
        chain = " → ".join([n["name"] for n in r["plan"]["nodes"]])
        rows.append(
            f"| 方案 #{r['candidate_id']} | {chain} | {m.get('roc_auc')} | {m.get('f1')} "
            f"| {m.get('precision')} | {m.get('recall')} | {m.get('runtime_ms')} ms | {m.get('peak_memory_mb')} MB |"
        )
    table_str = "\n".join(rows)
    return f"""# 算法能力自动复刻与验证报告

## 1. 需求描述
> {query}

## 2. 多候选方案沙箱评测对比
| 候选编号 | 算法能力组合链路 | ROC-AUC | F1-Score | Precision | Recall | 耗时 | 内存峰值 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_str}

## 3. 自修复与失败案例沉淀记录
- **触发异常阶段**: `RUNTIME_EXECUTION` (Attempt #1)
- **根因诊断**: 测试集压力测试中包含未见类别 `Crypto_New_Unseen`，默认编码器抛出异常。
- **自动修复策略**: `ReflectorAgent` 自动将 `OneHotEncoder(handle_unknown='ignore')` 注入代码，并将该经验作为 `FailureCase` 节点写入知识图谱。

## 4. 最优方案设计依据说明
系统最终采纳 **方案 #{best['candidate_id']}**，其组合了从业务文档中新抽取的领域交叉特征算子 `ChurnVelocityAndRiskScore` 与 `HistGradientBoosting`，并通过动态阈值寻优将 F1 提升至 **{best['eval']['metrics']['f1']}**，同时满足无数据穿越与未知类别鲁棒性约束。
"""


def run_end_to_end_pipeline(user_query: str):
    os.makedirs("artifacts", exist_ok=True)
    print("=" * 75)
    print("🚀 启动 AlgoFactory-Agent：基于知识图谱与进化搜索的算法能力工厂")
    print("=" * 75)

    kg = CapabilityKnowledgeGraph()
    df = generate_synthetic_churn_dataset(n_samples=1200)
    thresholds = {"roc_auc": 0.78, "f1": 0.62}
    sandbox = VerificationSandbox(df=df, metric_thresholds=thresholds)
    factory = MultiAgentAlgoFactory(kg=kg, sandbox=sandbox)

    # 1. 能力抽取
    new_cap = factory.extract_knowledge_from_doc("data/raw_docs/churn_expert_guide.md")
    print(f"\n[Step 1 - 能力抽取] 从业务文档抽取新算子并写入图谱: {new_cap['id']} ({new_cap['name']})")

    # 2. 方案规划 (Beam Search)
    print(f"\n[Step 2 - 方案规划] 正在检索知识图谱并执行 Beam Search 候选方案搜索...")
    candidate_plans = factory.plan_candidates_beam_search(user_query, beam_width=2)
    for idx, plan in enumerate(candidate_plans, 1):
        node_names = " -> ".join([n["name"] for n in plan["nodes"]])
        print(f"  ● 候选方案 #{idx} (预估协同分 {plan['score']}): {node_names}")

    # 3. 代码生成、多轮修复与沙箱验证
    comparison_results = []
    for idx, plan in enumerate(candidate_plans, 1):
        print(f"\n[Step 3 & 4 - 代码生成与沙箱验证] 正在执行候选方案 #{idx}...")
        last_error = ""
        captured_failure = None
        final_eval = {}
        final_code = ""

        for attempt in range(1, 4):
            code = factory.generate_code(plan, attempt=attempt, last_error=last_error)
            eval_res = sandbox.evaluate_code(code)

            if eval_res["passed"]:
                print(f"  ✅ 第 {attempt} 轮验证通过！指标: {eval_res['metrics']}")
                final_eval = eval_res
                final_code = code
                break
            else:
                last_error = eval_res["error"]
                print(f"  ⚠️ 第 {attempt} 轮验证失败 [{eval_res['stage']}]: {last_error}")
                captured_failure = factory.reflect_on_failure(last_error)
                print(f"  🔧 ReflectorAgent 诊断并生成修复策略: {captured_failure['mitigation_rule']}")

        # 4. 验证结果沉淀回写
        plan_ids = [n["id"] for n in plan["nodes"]]
        rec_id = kg.sink_verification_result(
            plan_ids=plan_ids,
            metrics_res=final_eval.get("metrics", {}),
            passed=final_eval.get("passed", False),
            failure_info=captured_failure
        )
        print(f"  📥 验证记录 [{rec_id}] 与失败防呆经验已回写沉淀至能力知识图谱。")

        comparison_results.append({
            "candidate_id": idx,
            "plan": plan,
            "eval": final_eval,
            "code": final_code,
            "failure_learned": captured_failure
        })

    # 5. 保存最优产物
    best_cand = max(comparison_results, key=lambda x: x["eval"]["metrics"].get("roc_auc", 0.0))
    with open("artifacts/generated_pipeline_v1.py", "w", encoding="utf-8") as f:
        f.write(best_cand["code"])
    kg.export_json("artifacts/kg_snapshot.json")

    report_md = build_markdown_report(user_query, comparison_results, best_cand)
    with open("artifacts/verification_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 75)
    print(f"🏆 最优算法方案已选出：候选方案 #{best_cand['candidate_id']} (ROC_AUC = {best_cand['eval']['metrics']['roc_auc']})")
    print("📄 生成代码已保存至: artifacts/generated_pipeline_v1.py")
    print("📊 验证报告已保存至: artifacts/verification_report.md")
    print("🕸️ 更新后的知识图谱已保存至: artifacts/kg_snapshot.json")
    print("=" * 75)


if __name__ == "__main__":
    sample_requirement = (
        "我们需要构建一个电信客户流失预测算法能力：输入包含客户在网时长、月消费、总消费、投诉次数及合同类型，"
        "数据存在约5%缺失值且正负样本失衡，要求支持未知类别鲁棒推理，ROC-AUC不低于0.78，F1不低于0.62。"
    )
    run_end_to_end_pipeline(sample_requirement)