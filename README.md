# AI-for-science

东北师范大学 · 刘炳卓（2025 级计算机类）本科科研申请相关项目代码与实验结果。

仓库地址：https://github.com/bzl094765-dotcom/AI-for-science

## 项目一览

| 项目 | 目录 | 任务 | 主要结果 |
|------|------|------|----------|
| 房价预测 | [`house-price-prediction/`](./house-price-prediction/) | 结构化数据回归（PyTorch MLP） | 测试集 **R² = 0.791**，RMSE ≈ 1399，MAE ≈ 750 |
| MNIST 手写数字识别 | [`mnist-digit-recognition/`](./mnist-digit-recognition/) | 图像分类（PyTorch CNN） | 验证 **99.16%**，测试 **99.07%** |

## 1. 房价预测（结构化回归）

路径：`house-price-prediction/`

- 脚本：`run_house_price_train.py`
- 结果：`result/metrics.json`、`training_results_noleak.png`、`feature_importance_noleak.csv`、模型权重

要点：
- 脏数据清洗、特征工程、训练/验证/测试划分
- **主动剔除由目标价格反推的 `price_per_sqft`，避免标签泄漏**
- 有效样本 161,492，特征 55 维

运行（需自备原始数据 CSV，并按脚本内路径放置）：

```bash
cd house-price-prediction
python run_house_price_train.py
```

## 2. MNIST 手写数字识别

路径：`mnist-digit-recognition/`

- 脚本：`train_mnist.py`
- 结果：`outputs/mnist_live_board.png`、`outputs/mnist_cnn_best.pth`

运行：

```bash
cd mnist-digit-recognition
python train_mnist.py --non-interactive
```

首次运行会自动下载 MNIST 到 `./data`。

## 环境依赖

- Python 3.9+
- PyTorch、torchvision
- NumPy、Pandas、scikit-learn、Matplotlib

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

## 说明

- 为控制仓库体积，**未上传原始大数据集**；实验结果与权重已归档。
- 本仓库用于本科科研申请材料展示，代码均为本人独立完成。
