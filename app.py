import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from kg.graph_store import CapabilityKnowledgeGraph
from sandbox.evaluator import VerificationSandbox, generate_synthetic_churn_dataset
from agents.workflow import MultiAgentAlgoFactory
from main import build_markdown_report


st.set_page_config(
    page_title="AlgoFactory-Agent 算法能力工厂",
    page_icon="⚙️",
    layout="wide"
)


def render_interactive_kg_html(kg: CapabilityKnowledgeGraph, height: int = 500) -> str:
    """将 NetworkX 知识图谱渲染为内嵌交互式拓扑图（零外部文件依赖，支持拖拽与悬停查看属性）"""
    nodes_data = []
    edges_data = []

    color_map = {
        "Scenario": "#FF6B6B",           # 红色：业务场景
        "Capability": "#4D96FF",         # 蓝色：算法能力算子
        "Metric": "#6BCB77",             # 绿色：评价指标
        "Constraint": "#FFD93D",         # 黄色：业务约束
        "VerificationRecord": "#9B5DE5", # 紫色：验证记录
        "FailureCase": "#FF922B"         # 橙色：沉淀的失败经验
    }

    for nid, attrs in kg.graph.nodes(data=True):
        label_type = attrs.get("label", "Unknown")
        display_name = attrs.get("name") or attrs.get("error_type") or nid
        ver = f" (v{attrs['version']})" if "version" in attrs else ""
        score = f" | 历史分:{attrs['historical_score']}" if "historical_score" in attrs else ""
        detail = json.dumps(attrs, ensure_ascii=False)
        nodes_data.append({
            "id": nid,
            "label": f"{display_name}{ver}",
            "group": label_type,
            "color": color_map.get(label_type, "#A0AEC0"),
            "title": f"[{label_type}] {nid}{score}\n{detail}"
        })

    for u, v, attrs in kg.graph.edges(data=True):
        rel = attrs.get("relation", "RELATED")
        synergy = f" ({attrs['synergy']})" if "synergy" in attrs else ""
        edges_data.append({
            "from": u,
            "to": v,
            "label": f"{rel}{synergy}"
        })

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        body {{ margin: 0; padding: 0; background-color: #0e1117; color: #fafafa; font-family: sans-serif; }}
        #mynetwork {{ width: 100%; height: {height}px; border: 1px solid #2d3748; border-radius: 8px; background: #161b22; }}
        .legend {{ display: flex; gap: 14px; padding: 8px 12px; font-size: 13px; flex-wrap: wrap; background: #1e2530; border-radius: 6px; margin-bottom: 8px; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
      </style>
    </head>
    <body>
      <div class="legend">
        <div class="legend-item"><span class="dot" style="background:#FF6B6B"></span>业务场景 (Scenario)</div>
        <div class="legend-item"><span class="dot" style="background:#4D96FF"></span>算法算子 (Capability)</div>
        <div class="legend-item"><span class="dot" style="background:#6BCB77"></span>评价指标 (Metric)</div>
        <div class="legend-item"><span class="dot" style="background:#FFD93D"></span>业务约束 (Constraint)</div>
        <div class="legend-item"><span class="dot" style="background:#9B5DE5"></span>验证记录 (Verification)</div>
        <div class="legend-item"><span class="dot" style="background:#FF922B"></span>失败避坑经验 (FailureCase)</div>
      </div>
      <div id="mynetwork"></div>
      <script type="text/javascript">
        const nodes = new vis.DataSet({json.dumps(nodes_data, ensure_ascii=False)});
        const edges = new vis.DataSet({json.dumps(edges_data, ensure_ascii=False)});
        const container = document.getElementById('mynetwork');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
          nodes: {{
            shape: 'dot',
            size: 20,
            font: {{ size: 13, color: '#e2e8f0', strokeWidth: 2, strokeColor: '#0f172a' }},
            borderWidth: 2
          }},
          edges: {{
            width: 1.5,
            color: {{ color: '#64748b', highlight: '#38bdf8' }},
            arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
            font: {{ size: 10, color: '#94a3b8', align: 'middle', strokeWidth: 0 }},
            smooth: {{ type: 'cubicBezier', forceDirection: 'none', roundness: 0.4 }}
          }},
          physics: {{
            barnesHut: {{ gravitationalConstant: -3800, centralGravity: 0.3, springLength: 130 }}
          }},
          interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    return html_code


# 初始化 Session State
if "kg" not in st.session_state:
    st.session_state.kg = CapabilityKnowledgeGraph()
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None


# 顶部标题栏
st.title("⚙️ AlgoFactory-Agent：基于知识图谱与进化搜索的算法能力工厂")
st.markdown(
    "实现 **能力抽取 → GraphRAG + Beam Search 方案规划 → 算法代码自动复刻 → AST 沙箱验证与多轮自修复 → 知识图谱沉淀** 端到端闭环。"
)

# 侧边栏：验证配置与知识抽取操作
with st.sidebar:
    st.header("🛠️ 验证配置与任务参数")
    scenario_name = st.selectbox("选择行业场景", ["电信客户流失预测 (SC_CHURN)"])
    auc_thresh = st.slider("ROC-AUC 达标阈值", min_value=0.65, max_value=0.90, value=0.78, step=0.01)
    f1_thresh = st.slider("F1-Score 达标阈值", min_value=0.50, max_value=0.80, value=0.62, step=0.01)
    beam_width = st.radio("Beam Search 候选方案宽度 (Top-K)", [1, 2], index=1, horizontal=True)
    n_samples = st.number_input("沙箱测试样本规模", min_value=400, max_value=3000, value=1200, step=200)

    st.divider()
    st.subheader("📚 能力知识库管理")
    if st.button("🔄 重置初始知识图谱", use_container_width=True):
        st.session_state.kg = CapabilityKnowledgeGraph()
        st.session_state.pipeline_result = None
        st.rerun()

# 主界面左侧输入与启动区
default_req = (
    "我们需要构建一个电信客户流失预测算法能力：输入包含客户在网时长、月消费、总消费、投诉次数及合同类型，"
    "数据存在约5%缺失值且正负样本失衡，要求支持未知类别鲁棒推理，ROC-AUC不低于0.78，F1不低于0.62。"
)
user_req = st.text_area("📝 输入自然语言算法能力描述：", value=default_req, height=90)

run_btn = st.button("🚀 一键启动：能力抽取 → 方案搜索 → 代码生成 → 沙箱自修复验证 → 图谱沉淀", type="primary", use_container_width=True)

if run_btn:
    os.makedirs("artifacts", exist_ok=True)
    kg: CapabilityKnowledgeGraph = st.session_state.kg
    df = generate_synthetic_churn_dataset(n_samples=int(n_samples))
    thresholds = {"roc_auc": float(auc_thresh), "f1": float(f1_thresh)}
    sandbox = VerificationSandbox(df=df, metric_thresholds=thresholds)
    factory = MultiAgentAlgoFactory(kg=kg, sandbox=sandbox)

    execution_logs = []

    with st.status("🤖 多智能体工作流正在执行...", expanded=True) as status:
        st.write("**[Step 1 - ExtractorAgent]** 正在从 `data/raw_docs/churn_expert_guide.md` 抽取领域算子...")
        new_cap = factory.extract_knowledge_from_doc("data/raw_docs/churn_expert_guide.md")
        execution_logs.append(f"✅ 成功从业务文档抽取新特征算子 `{new_cap['id']}` ({new_cap['name']}) 并入库。")

        st.write(f"**[Step 2 - PlannerAgent]** 正在检索子图并执行 Beam Search (width={beam_width})...")
        plans = factory.plan_candidates_beam_search(user_req, beam_width=int(beam_width))
        for idx, p in enumerate(plans, 1):
            chain = " → ".join([n["name"] for n in p["nodes"]])
            execution_logs.append(f"🔍 规划候选方案 #{idx} (预估协同分 `{p['score']}`): {chain}")

        comparison_results = []
        for idx, plan in enumerate(plans, 1):
            st.write(f"**[Step 3 & 4 - Coder & Sandbox]** 正在复刻并验证候选方案 #{idx}...")
            last_error = ""
            captured_failure = None
            final_eval = {}
            final_code = ""
            attempt_trace = []

            for attempt in range(1, 4):
                code = factory.generate_code(plan, attempt=attempt, last_error=last_error)
                eval_res = sandbox.evaluate_code(code)
                attempt_trace.append({"attempt": attempt, "res": eval_res, "code": code})

                if eval_res["passed"]:
                    execution_logs.append(
                        f"  ✅ 方案 #{idx} 在第 {attempt} 轮验证通过！指标: AUC={eval_res['metrics']['roc_auc']}, F1={eval_res['metrics']['f1']}"
                    )
                    final_eval = eval_res
                    final_code = code
                    break
                else:
                    last_error = eval_res["error"]
                    captured_failure = factory.reflect_on_failure(last_error)
                    execution_logs.append(
                        f"  ⚠️ 方案 #{idx} 第 {attempt} 轮沙箱拦截 [{eval_res['stage']}]: `{last_error}`"
                    )
                    execution_logs.append(
                        f"  🔧 ReflectorAgent 诊断并注入自修复规则: *{captured_failure['mitigation_rule']}*"
                    )

            plan_ids = [n["id"] for n in plan["nodes"]]
            rec_id = kg.sink_verification_result(
                plan_ids=plan_ids,
                metrics_res=final_eval.get("metrics", {}),
                passed=final_eval.get("passed", False),
                failure_info=captured_failure
            )
            execution_logs.append(f"  📥 验证记录 `{rec_id}` 与失败案例节点已回写沉淀至知识图谱（算子版本号已自动更新）。")

            comparison_results.append({
                "candidate_id": idx,
                "plan": plan,
                "eval": final_eval,
                "code": final_code,
                "failure_learned": captured_failure,
                "trace": attempt_trace
            })

        best_cand = max(comparison_results, key=lambda x: x["eval"]["metrics"].get("roc_auc", 0.0))
        with open("artifacts/generated_pipeline_v1.py", "w", encoding="utf-8") as f:
            f.write(best_cand["code"])
        kg.export_json("artifacts/kg_snapshot.json")
        report_md = build_markdown_report(user_req, comparison_results, best_cand)
        with open("artifacts/verification_report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        # 保存可视化 HTML 到 artifacts 供离线查阅
        vis_html = render_interactive_kg_html(kg, height=600)
        with open("artifacts/kg_visualization.html", "w", encoding="utf-8") as f:
            f.write(vis_html)

        st.session_state.pipeline_result = {
            "logs": execution_logs,
            "comparisons": comparison_results,
            "best": best_cand,
            "report_md": report_md
        }
        status.update(label="🎉 算法能力自动复刻、验证与图谱沉淀全闭环完成！", state="complete", expanded=False)


# 多标签页结果展示区
tab_kg, tab_comp, tab_code, tab_report = st.tabs([
    "🕸️ 能力知识图谱可视化",
    "📊 多候选方案评测与自修复轨迹",
    "💻 生成的最优算法代码",
    "📄 自动验证报告"
])

with tab_kg:
    st.subheader("实时行业算法能力知识图谱（支持鼠标滚轮缩放、拖拽节点、悬停查看属性与版本号）")
    kg_obj: CapabilityKnowledgeGraph = st.session_state.kg
    col1, col2, col3 = st.columns(3)
    col1.metric("图谱节点总数", len(kg_obj.graph.nodes))
    col2.metric("语义关系边总数", len(kg_obj.graph.edges))
    cap_cnt = sum(1 for _, d in kg_obj.graph.nodes(data=True) if d.get("label") == "Capability")
    col3.metric("已沉淀算法算子数", cap_cnt)

    html_content = render_interactive_kg_html(kg_obj, height=520)
    components.html(html_content, height=570, scrolling=False)

    with st.expander("查看算子节点当前版本与历史得分明细表"):
        cap_rows = []
        for nid, d in kg_obj.graph.nodes(data=True):
            if d.get("label") == "Capability":
                cap_rows.append({
                    "算子ID": nid,
                    "名称": d.get("name"),
                    "类别": d.get("category"),
                    "当前版本": d.get("version"),
                    "综合历史得分": d.get("historical_score"),
                    "适用条件": d.get("applicable_conditions")
                })
        st.dataframe(pd.DataFrame(cap_rows), use_container_width=True)

with tab_comp:
    res = st.session_state.pipeline_result
    if not res:
        st.info("请点击上方 **「🚀 一键启动」** 按钮运行工作流，查看多候选方案对比与多轮代码自修复日志。")
    else:
        st.subheader("🏆 多候选方案沙箱竞技排行榜")
        table_data = []
        for r in res["comparisons"]:
            m = r["eval"]["metrics"]
            table_data.append({
                "候选方案": f"方案 #{r['candidate_id']}" + (" ⭐ (最优采纳)" if r["candidate_id"] == res["best"]["candidate_id"] else ""),
                "算子组合链路": " → ".join([n["name"] for n in r["plan"]["nodes"]]),
                "图谱预估协同分": r["plan"]["score"],
                "ROC-AUC": m.get("roc_auc"),
                "F1-Score": m.get("f1"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "运行耗时 (ms)": m.get("runtime_ms"),
                "内存峰值 (MB)": m.get("peak_memory_mb")
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        st.subheader("🔧 多智能体执行与多轮自我修复 (Self-Healing) 详细日志")
        for log_line in res["logs"]:
            st.markdown(f"- {log_line}")

with tab_code:
    res = st.session_state.pipeline_result
    if not res:
        st.info("运行工作流后将在此展示自动生成的标准算法代码。")
    else:
        st.subheader(f"最优方案 (#{res['best']['candidate_id']}) 生成的可运行代码 (`artifacts/generated_pipeline_v1.py`)")
        st.code(res["best"]["code"], language="python")

with tab_report:
    res = st.session_state.pipeline_result
    if not res:
        st.info("运行工作流后将在此展示自动生成的 Markdown 验证报告。")
    else:
        st.markdown(res["report_md"])