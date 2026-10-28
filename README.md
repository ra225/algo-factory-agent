# AlgoFactory-Agent：基于知识图谱与进化搜索的算法能力工厂

## 1. 项目背景和目标
在真实行业算法研发中，数据预处理策略、领域特征工程、模型选型经验与失败避坑指南往往散落在非结构化文档与历史代码库中。本项目设计并实现了 **AlgoFactory-Agent** 原型系统，打通了**“非结构化能力抽取 → 知识图谱结构化存储 → GraphRAG + Beam Search 方案规划 → 代码自动复刻 → AST 安全沙箱验证 → 失败归因自修复与知识回写沉淀”**的完整闭环。

## 2. 系统架构和模块设计
系统分为五层核心模块：
1. **知识抽取层 (`ExtractorAgent`)**：从业务文档、实验记录及历史代码仓库 AST 中抽取算子能力、适用条件与业务约束。
2. **能力知识图谱层 (`CapabilityKnowledgeGraph`)**：基于属性图存储场景、算子、指标、约束、验证记录与失败案例，支持能力版本号自动演进。
3. **搜索规划层 (`PlannerAgent`)**：结合子图检索与 **Beam Search（束搜索）**，根据算子历史表现与算子间协同权重（`synergy`）规划 Top-K 候选算法链路。
4. **代码复刻与修复层 (`CoderAgent` & `ReflectorAgent`)**：生成符合统一接口契约的可运行代码，并根据沙箱报错进行多轮迭代修复。
5. **契约沙箱与沉淀层 (`VerificationSandbox`)**：涵盖 AST 静态安全检查、未见类别压力测试、多指标评估、耗时/内存探针及图谱自动回写。

## 3. 能力知识图谱 Schema 和示例
- **节点类型**：`Scenario`（业务场景）、`Capability`（算法能力算子）、`Metric`（评价指标）、`Constraint`（业务与工程约束）、`VerificationRecord`（验证记录）、`FailureCase`（失败经验节点）。
- **边关系类型**：`APPLICABLE_TO`、`COMPOSED_WITH`（含协同权重）、`REQUIRES_METRIC`、`HAS_CONSTRAINT`、`VERIFIED_BY`、`HAS_FAILURE_EXPERIENCE`。
- **导出文件**：运行后自动生成 `artifacts/kg_snapshot.json`。

## 4. Agent 工作流设计
系统采用多智能体状态机协同：
`用户输入需求` → `Extractor 抽取增量知识` → `Planner 执行 GraphRAG 与 Beam Search 输出候选方案集` → `Coder 生成候选代码` → `Sandbox 执行 AST 审计与测试集压力评估` → （若失败）`Reflector 诊断根因、提炼避坑规则并指导 Coder 重写` → `择优输出代码并将指标与失败节点回写至知识图谱`。

## 5. 环境配置和运行方法
```bash
# 1. 创建虚拟环境并安装依赖
pip install -r requirements.txt

# 2. (可选) 配置大模型 API Key；若不配置，系统自动启用内置确定性仿真引擎确保 100% 本地复现
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="[https://api.openai.com/v1](https://api.openai.com/v1)"

# 3. 一键运行端到端核心流程（抽取 -> 规划 -> 生成 -> 修复 -> 验证 -> 沉淀）
python main.py

# 4. 运行单元测试
pytest tests/

# 5. 启动交互式 Web 界面与知识图谱可视化看板
python -m streamlit run app.py
```
