# 算法能力自动复刻与验证报告

## 1. 需求描述
> 我们需要构建一个电信客户流失预测算法能力：输入包含客户在网时长、月消费、总消费、投诉次数及合同类型，数据存在约5%缺失值且正负样本失衡，要求支持未知类别鲁棒推理，ROC-AUC不低于0.78，F1不低于0.62。

## 2. 多候选方案沙箱评测对比
| 候选编号 | 算法能力组合链路 | ROC-AUC | F1-Score | Precision | Recall | 耗时 | 内存峰值 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 方案 #1 | RobustImputerEncoder → ChurnVelocityAndRiskScore → HistGradientBoosting → F1OptimalThresholdTuner | 0.8666 | 0.6842 | 0.6933 | 0.6753 | 576.81 ms | 1.018 MB |
| 方案 #2 | RobustImputerEncoder → ChurnVelocityAndRiskScore → BalancedRandomForest → F1OptimalThresholdTuner | 0.8617 | 0.6424 | 0.6023 | 0.6883 | 957.65 ms | 0.406 MB |

## 3. 自修复与失败案例沉淀记录
- **触发异常阶段**: `RUNTIME_EXECUTION` (Attempt #1)
- **根因诊断**: 测试集压力测试中包含未见类别 `Crypto_New_Unseen`，默认编码器抛出异常。
- **自动修复策略**: `ReflectorAgent` 自动将 `OneHotEncoder(handle_unknown='ignore')` 注入代码，并将该经验作为 `FailureCase` 节点写入知识图谱。

## 4. 最优方案设计依据说明
系统最终采纳 **方案 #1**，其组合了从业务文档中新抽取的领域交叉特征算子 `ChurnVelocityAndRiskScore` 与 `HistGradientBoosting`，并通过动态阈值寻优将 F1 提升至 **0.6842**，同时满足无数据穿越与未知类别鲁棒性约束。
