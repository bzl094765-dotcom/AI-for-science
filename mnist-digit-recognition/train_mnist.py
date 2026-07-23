"""
MNIST 手写数字识别（PyTorch + Matplotlib 实时交互画板）

运行：
    python train_mnist.py

说明：
- 训练过程中会弹出 Matplotlib 窗口，实时刷新损失与准确率曲线
- 关闭窗口不会立刻停训；想中断请在终端 Ctrl+C
- 首次运行会自动下载 MNIST 到 ./data
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 解决 Windows Conda 环境下 OpenMP 重复初始化导致崩溃的问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if "--non-interactive" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ========================= 可调参数 =========================
BATCH_SIZE = 128
EPOCHS = 8
LR = 1e-3
NUM_WORKERS = 0  # Windows 下建议先用 0，避免多进程问题
DATA_DIR = Path(__file__).resolve().parent / "data"
SAVE_DIR = Path(__file__).resolve().parent / "outputs"
# ===========================================================


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class LiveBoard:
    """Matplotlib 实时交互画板：损失 + 准确率 + 样本预测。"""

    def __init__(self) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=(12, 7))
        self.ax_loss = self.fig.add_subplot(2, 2, 1)
        self.ax_acc = self.fig.add_subplot(2, 2, 2)
        self.ax_samples = self.fig.add_subplot(2, 1, 2)

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []

        self.fig.canvas.manager.set_window_title("MNIST 实时训练看板")
        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        sample_imgs: torch.Tensor | None = None,
        sample_preds: list[int] | None = None,
        sample_labels: list[int] | None = None,
    ) -> None:
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

        epochs = list(range(1, len(self.train_losses) + 1))

        self.ax_loss.clear()
        self.ax_loss.plot(epochs, self.train_losses, label="Train Loss")
        self.ax_loss.plot(epochs, self.val_losses, label="Val Loss")
        self.ax_loss.set_title(f"Loss (Epoch {epoch})")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend(loc="upper right")
        self.ax_loss.grid(True, alpha=0.3)

        self.ax_acc.clear()
        self.ax_acc.plot(epochs, self.train_accs, label="Train Acc")
        self.ax_acc.plot(epochs, self.val_accs, label="Val Acc")
        self.ax_acc.set_title(f"Accuracy (Val={val_acc:.2f}%)")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.ax_acc.set_ylim(0, 100)
        self.ax_acc.legend(loc="lower right")
        self.ax_acc.grid(True, alpha=0.3)

        self.ax_samples.clear()
        self.ax_samples.set_title("Validation Samples (pred / true)")
        self.ax_samples.axis("off")
        if sample_imgs is not None and sample_preds is not None and sample_labels is not None:
            n = min(10, sample_imgs.size(0))
            # 拼成一条带状图，便于一眼看预测对错
            row = sample_imgs[:n, 0].cpu()
            canvas = torch.cat([row[i] for i in range(n)], dim=1).numpy()
            self.ax_samples.imshow(canvas, cmap="gray")
            titles = [f"{p}/{t}" for p, t in zip(sample_preds[:n], sample_labels[:n])]
            self.ax_samples.set_xlabel("  |  ".join(titles), fontsize=9)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"看板已保存: {path}")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    sample_imgs = None
    sample_preds = None
    sample_labels = None

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += images.size(0)

        if batch_idx == 0:
            sample_imgs = images.detach().cpu()
            sample_preds = preds.detach().cpu().tolist()
            sample_labels = labels.detach().cpu().tolist()

    return (
        total_loss / max(total, 1),
        100.0 * total_correct / max(total, 1),
        sample_imgs,
        sample_preds,
        sample_labels,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), 100.0 * total_correct / max(total, 1)


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]
    train_set = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

    # 从训练集划出验证集，测试集最后单独报告
    val_size = 5000
    train_size = len(train_set) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    board = LiveBoard()

    best_val_acc = -1.0
    best_path = SAVE_DIR / "mnist_cnn_best.pth"
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc, imgs, preds, labels = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} acc={val_acc:.2f}%"
        )

        board.update(epoch, train_loss, val_loss, train_acc, val_acc, imgs, preds, labels)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                },
                best_path,
            )

    # 加载最佳模型做测试集评估
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, _, _, _ = evaluate(model, test_loader, device, criterion)

    board.save(SAVE_DIR / "mnist_live_board.png")
    elapsed = time.time() - start

    print("\n======== 最终结果 ========")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (epoch {ckpt['epoch']})")
    print(f"测试集准确率:   {test_acc:.2f}%")
    print(f"测试集损失:     {test_loss:.4f}")
    print(f"耗时:           {elapsed:.1f}s")
    print(f"模型已保存:     {best_path}")
    print(f"看板图片已保存: {SAVE_DIR / 'mnist_live_board.png'}")

    import sys
    plt.ioff()
    if "--non-interactive" in sys.argv:
        print("（非交互运行模式：看板图片已保存，自动退出）")
        plt.close("all")
    else:
        print("看板窗口已保持打开，请直接关闭画板窗口以结束程序。")
        try:
            plt.show(block=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
