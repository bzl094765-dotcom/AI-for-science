# 房价预测（PyTorch）

端到端结构化数据回归：清洗 → 特征工程 → MLP 训练 → 评估。

## 实测结果（无标签泄漏设定）

见 `result/metrics.json`：

- samples: 161492
- n_features: 55
- test R²: 0.791
- RMSE ≈ 1399, MAE ≈ 750

## 文件

- `run_house_price_train.py`：训练脚本
- `result/`：指标、曲线、特征重要性、模型权重
