# AlgoFactory-Agent：基于知识图谱与进化搜索的算法能力工厂

## 1. 项目背景和目标

在真实行业算法研发中，数据预处理策略、领域特征工程、模型选型经验与失败避坑指南往往散落在非结构化文档与历史代码库中。本项目设计并实现了 **AlgoFactory-Agent** 原型系统，打通了 **非结构化能力抽取 → 知识图谱结构化存储 → GraphRAG + Beam Search 方案规划 → 代码自动复刻 → AST 安全沙箱验证 → 失败归因自修复与知识回写沉淀** 的完整闭环。

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

## 6. 示例数据和测试任务说明
- **测试场景**：电信客户流失预测（`SC_CHURN`），属于二分类表格任务。
- **特征构成**：数值特征（在网时长 `tenure`、月消费 `monthly_charges`、总消费 `total_charges`、投诉次数 `support_calls`）与类别特征（合同类型 `contract_type`、支付方式 `payment_method`）。
- **沙箱注入的工程挑战**：
  1. **数据缺陷**：随机注入 5% 缺失值（`NaN`），且正负样本严重失衡（约 1:3）。
  2. **非线性依赖**：通过逻辑斯蒂隐函数构造了复杂的交叉特征组合逻辑。
  3. **线上边界压力测试**：在沙箱测试集（`X_test_stress`）中故意篡改了一条数据的支付方式为 `Crypto_New_Unseen`（训练集中未曾出现的新类别），以此检验生成代码的线上鲁棒性。

## 7. 生成算法代码示例
生成的完整代码输出在 `artifacts/generated_pipeline_v1.py` 中。生成的代码强制实现 `GeneratedCapability` 类契约，其核心亮点包括：
- **特征防崩**：在自定义特征构造前，利用 `.fillna(1.0).clip(lower=1.0)` 进行数学分母安全兜底。
- **数据防穿越**：严格封装 `ColumnTransformer` 和 `Pipeline`，区分 `fit_transform` 与 `transform`。
- **动态阈值寻优**：基于训练概率输出 `predict_proba` 遍历搜索，自动将 F1-Score 从 0.5 默认切分的低谷拉升至达标线以上。

```python
# 核心片段节选：自修复后的编码器与分类器组合
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # 此处 handle_unknown='ignore' 为 Agent 第二轮自动吸收教训后注入的修正
    ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
# 选型采用了图谱规划的 GBDT 模型应对非线性特征
self.model = HistGradientBoostingClassifier(
    max_iter=100, max_depth=4, learning_rate=0.05, class_weight='balanced'
)
```
## 8. 验证结果和报告样例
运行后会自动生成 `artifacts/verification_report.md`。沙箱对 Beam Search 规划出的两条候选链路进行了真实数据的赛马对比：
- **方案 #1 (GBDT)**：结合了 `ChurnVelocityAndRiskScore` 特征，ROC-AUC 达 0.86+，F1 达 0.68+，耗时 ~650ms。
- **方案 #2 (Random Forest)**：同样的特征工程，但由于随机森林在当前非线性组合下 Precision 较低，导致 F1 指标落后，且运行耗时超过 1.3 秒。
系统最终自动采纳 **方案 #1**，并将其指标与新的版本号（如 `v1.0.2`）更新至图谱节点。

## 9. 遇到的挑战和解决方案
1. **挑战：生成代码的安全性与接口一致性问题**
   - **解决方案**：引入 `ASTSecurityAndContractChecker`。在执行 `exec()` 前，通过抽象语法树（AST）静态拦截 `os`/`subprocess` 等危险系统调用，并强制校验生成的类是否包含 `fit`/`predict` 等标准契约方法。
2. **挑战：LLM 容易重复犯错，无长期记忆**
   - **解决方案**：设计了**“沙箱拦截 → Reflector 归因 → 知识图谱沉淀”**的免疫机制。测试集抛出未见类别异常时，系统提取防呆规则并存为 `FailureCase` 图谱节点。下次规划同类算子时，该节点作为前置 Prompt 被检索注入，实现 Agent 的“吃一堑长一智”。
3. **挑战：单次生成的局部最优困境**
   - **解决方案**：放弃单纯的单次 Prompt 问答，引入基于图谱的 **Beam Search (束搜索)**。结合算子的“单体历史得分”与图谱边的“协同默契度 (synergy)”，并行规划出 Top-K 条候选链路放入沙箱进行末位淘汰。

## 10. 后续可扩展方向
1. **多级沙箱与容器化隔离**：将目前的 AST + 本地 `subprocess` 评估模式，升级为基于 Docker 或 Firecracker 微虚拟机的全隔离物理沙箱，支持更复杂的系统级库安装。
2. **MCTS 与演化搜索接入**：在流水线规划层，用蒙特卡洛树搜索（MCTS）替代 Beam Search，用于更大规模的超参数拓扑空间寻优。
3. **企业代码库资产挖掘冷启动**：利用 AST 批量解析企业内部历史 Git 仓库，自动抽取常用特征工程函数，全自动构建知识图谱冷启动数据。
4. **多模态与跨场景泛化**：当前图谱 Schema 天然支持横向扩展。未来只需往 `Scenario` 节点注入 NLP/CV 场景约束，即可零代码改动平滑迁移至文本分类或图像检测任务。
