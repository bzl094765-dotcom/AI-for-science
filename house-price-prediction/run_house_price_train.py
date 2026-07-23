"""
房屋价格预测训练脚本（可跑通版）

基于 DL-x-x4.2.py 逻辑，修复：
1. 数据路径改为本仓库 dataset/house_prices.csv
2. 去掉用目标价格构造的 price_per_sqft（避免标签泄漏）
3. 无界面保存结果到 result/，不阻塞 input/plt.show
4. 训练结束后写出 metrics.json 供简历填写

运行：
    python run_house_price_train.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "dataset" / "house_prices.csv"
RESULT_DIR = SCRIPT_DIR / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 训练量较大时可用子集先验证流程；正式跑请设为 None
MAX_ROWS = None  # 例如 30000；正式全量设 None


def extract_number(value):
    if pd.isna(value) or value == "":
        return np.nan

    value_str = str(value).strip()

    if "out of" in value_str.lower():
        match = re.search(r"(\d+)", value_str)
        return float(match.group(1)) if match else np.nan

    if "lac" in value_str.lower() or "lakh" in value_str.lower():
        numbers = re.findall(r"[\d.]+", value_str)
        if numbers:
            return float(numbers[0]) * 100000

    if any(
        unit in value_str.lower()
        for unit in ["sqft", "sq.ft", "sq ft", "sq. ft", "sq. foot", "sq foot"]
    ):
        numbers = re.findall(r"[\d.]+", value_str)
        if numbers:
            return float(numbers[0])

    try:
        return float(re.search(r"[\d.]+", value_str).group())
    except Exception:
        return np.nan


class HousePriceDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_factor=0.01):
        self.X = X
        self.y = y
        self.augment = augment
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = np.float32(self.y[idx])
        if self.augment:
            x = x + np.random.normal(0, self.noise_factor, x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(y)


class HousePriceModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def load_and_preprocess():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

    print(f"加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, header=0, dtype=str)
    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()
        print(f"使用子集前 {MAX_ROWS} 行做快速验证")

    print(f"原始形状: {df.shape}")

    columns = [
        "Index",
        "Title",
        "Description",
        "Amount(in rupees)",
        "Price (in rupees)",
        "location",
        "Carpet Area",
        "Status",
        "Floor",
        "Transaction",
        "Furnishing",
        "facing",
        "overlooking",
        "Society",
        "Bathroom",
        "Balcony",
        "Car Parking",
        "Ownership",
        "Super Area",
        "Dimensions",
        "Plot Area",
    ]
    # 兼容列数不一致
    rename = {c: columns[i] for i, c in enumerate(df.columns) if i < len(columns)}
    df = df.rename(columns=rename)

    for col in ["Index", "Title", "Description", "Amount(in rupees)", "Dimensions"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df["Price (in rupees)"] = df["Price (in rupees)"].apply(extract_number)
    df = df[df["Price (in rupees)"] > 0].dropna(subset=["Price (in rupees)"])
    price_upper = df["Price (in rupees)"].quantile(0.95)
    df = df[df["Price (in rupees)"] <= price_upper]

    numerical_columns = [
        "Carpet Area",
        "Bathroom",
        "Balcony",
        "Car Parking",
        "Super Area",
        "Plot Area",
    ]
    for col in numerical_columns:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(extract_number)
        if df[col].count() > 0:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0
        df[col] = df[col].fillna(median_val)

    df = df[df["Carpet Area"] > 0].copy()

    # 注意：不再用 Price 构造 price_per_sqft，避免标签泄漏
    if "Bathroom" in df.columns:
        df["bathroom_per_sqft"] = df["Bathroom"] / (df["Carpet Area"] + 1e-6)
    if "Plot Area" in df.columns and df["Plot Area"].sum() > 0:
        df["living_lot_ratio"] = df["Carpet Area"] / (df["Plot Area"] + 1e-6)

    categorical_columns = [
        "location",
        "Status",
        "Transaction",
        "Furnishing",
        "facing",
        "overlooking",
        "Society",
        "Ownership",
    ]
    feature_columns = [
        c
        for c in [
            "Carpet Area",
            "Bathroom",
            "Balcony",
            "Car Parking",
            "Super Area",
            "Plot Area",
            "bathroom_per_sqft",
            "living_lot_ratio",
        ]
        if c in df.columns
    ]

    for cat_col in categorical_columns:
        if cat_col not in df.columns:
            continue
        df[cat_col] = df[cat_col].fillna("Unknown")
        top_categories = df[cat_col].value_counts().nlargest(8).index.tolist()
        for category in top_categories:
            cat_col_name = f"{cat_col}_{str(category).replace(' ', '_').replace('/', '_')}"
            cat_col_name = re.sub(r"[^a-zA-Z0-9_]", "", cat_col_name)
            df[cat_col_name] = (df[cat_col] == category).astype(int)
            feature_columns.append(cat_col_name)

    existing_features = [c for c in feature_columns if c in df.columns]
    X = df[existing_features].copy()
    y = df["Price (in rupees)"].copy()

    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    print(f"清洗后样本: {X.shape[0]}, 特征维: {X.shape[1]}")
    return X, y, existing_features


def train_model(model, train_loader, val_loader, device, epochs=120, patience=15):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-7
    )

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target.view(-1, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target.view(-1, 1)).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter >= patience or epoch == epochs - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{epochs}: train={train_loss:.6f} val={val_loss:.6f} lr={lr:.2e}"
            )

        if patience_counter >= patience:
            print(f"早停触发（patience={patience}）")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for data, target in loader:
        data = data.to(device)
        output = model(data)
        preds.extend(output.cpu().numpy().flatten())
        targets.extend(target.numpy().flatten())

    preds = np.expm1(np.array(preds))
    targets = np.expm1(np.array(targets))
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds))
    return preds, targets, rmse, mae, r2


def pick_device() -> torch.device:
    """优先 GPU；若当前 PyTorch 不支持该显卡架构，自动回退 CPU。"""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as e:
        print(f"CUDA 不可用，回退 CPU。原因: {type(e).__name__}: {e}")
        return torch.device("cpu")


def main():
    t0 = time.time()
    X, y, feature_names = load_and_preprocess()

    y_log = np.log1p(y.values.astype(np.float64))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(np.float32))

    X_tv, X_test, y_tv, y_test = train_test_split(
        X_scaled, y_log, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15, random_state=42
    )

    print(f"划分: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    device = pick_device()
    print(f"设备: {device}")

    # CPU 用更大 batch 加速；GPU 用中等 batch
    if device.type == "cpu":
        batch_size = min(1024, max(64, len(X_train) // 50))
    else:
        batch_size = min(128, max(16, len(X_train) // 20))
    print(f"batch_size={batch_size}")
    train_loader = DataLoader(
        HousePriceDataset(X_train, y_train, augment=True),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        HousePriceDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        HousePriceDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    model = HousePriceModel(input_size=X_train.shape[1]).to(device)
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device, epochs=120, patience=15
    )

    preds, actuals, rmse, mae, r2 = evaluate(model, test_loader, device)
    print("\n======== 测试集指标 ========")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"R²:   {r2:.4f}")

    # 保存图表
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    max_val = np.percentile(actuals, 95) * 1.2
    mask = (actuals <= max_val) & (preds <= max_val)
    ax2.scatter(actuals[mask], preds[mask], s=8, alpha=0.5)
    lo, hi = actuals[mask].min(), actuals[mask].max()
    ax2.plot([lo, hi], [lo, hi], "r--")
    ax2.set_title(f"Pred vs Actual (R²={r2:.4f})")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = RESULT_DIR / "training_results_noleak.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # 特征重要性（首层权重）
    first_linear = next(m for m in model.model if isinstance(m, nn.Linear))
    importance = np.mean(np.abs(first_linear.weight.detach().cpu().numpy()), axis=0)
    fi = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_path = RESULT_DIR / "feature_importance_noleak.csv"
    fi.to_csv(fi_path, index=False)

    model_path = RESULT_DIR / "house_price_model_noleak.pth"
    torch.save(model.state_dict(), model_path)

    metrics = {
        "samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "train": int(len(X_train)),
        "val": int(len(X_val)),
        "test": int(len(X_test)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "top_features": fi.head(8).to_dict(orient="records"),
        "elapsed_sec": round(time.time() - t0, 1),
        "device": str(device),
        "note": "未使用由 Price 构造的 price_per_sqft，避免标签泄漏",
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n指标已保存: {metrics_path}")
    print(f"模型已保存: {model_path}")
    print(f"图表已保存: {fig_path}")
    print(f"总耗时: {metrics['elapsed_sec']}s")
    print("Top features:")
    print(fi.head(8).to_string(index=False))


if __name__ == "__main__":
    # Windows 控制台尽量 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
