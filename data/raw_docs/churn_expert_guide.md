# 电信客户流失预测（Churn Prediction）算法专家经验与实验报告

## 1. 业务背景与数据特征
输入数据包含客户在网时长（`tenure`）、月消费额（`monthly_charges`）、总消费额（`total_charges`）、客服投诉次数（`support_calls`）、合同类型（`contract_type`）与支付方式（`payment_method`）。
数据痛点：存在约 5% 缺失值，正负样本比例约 1:3（类别不平衡）。

## 2. 历史高价值特征工程策略
实验表明，单纯使用原始数值特征效果有限。构造以下两个领域交叉特征可将 AUC 提升 3% 以上：
- **消费速率偏差比（`charge_velocity`）**：`total_charges / (tenure * monthly_charges + 1.0)`，识别近期消费异动。
- **单位时长投诉密度（`support_density`）**：`support_calls / tenure`，衡量客户不满情绪强度。

## 3. 历史失败案例与工程约束
- **防数据泄露约束**：所有缺失值填充与编码器必须在训练集 `fit`，严禁在全量数据上提前 `fit`。
- **线上未知类别崩溃案例**：生产环境曾出现新型支付方式（如 `Crypto_New_Unseen`），导致默认 `OneHotEncoder` 抛出 `ValueError`，必须显式配置 `handle_unknown='ignore'`。