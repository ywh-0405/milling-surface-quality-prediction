"""
铣削表面质量预测框架 - 完整单文件版本
======================================
直接运行: python milling_prediction_framework.py

功能：
  1. 生成模拟铣削数据集（切削参数 + 三向振动 → Ra + 表面起伏频域）
  2. 训练多任务神经网络（共享编码器 + 双预测头）
  3. 评估 & 可视化（训练曲线/散点图/频域对比/综合Dashboard）
  4. 单样本预测接口演示
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║                      依赖库                                  ║
# ╚══════════════════════════════════════════════════════════════╝
import os, warnings
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════╗
# ║                   全局配置                                   ║
# ╚══════════════════════════════════════════════════════════════╝
NUM_FLUTES          = 4        # 四刃铣刀
FS                  = 5000     # 振动采样率 (Hz)
SIGNAL_DURATION     = 0.5      # 信号时长 (s)
N_FREQ_BINS         = 8        # 表面起伏频域分箱数

BATCH_SIZE          = 32
EPOCHS              = 150
LEARNING_RATE       = 1e-3
WEIGHT_RA           = 1.0      # Ra 任务损失权重
WEIGHT_WAVINESS     = 0.8      # 表面起伏任务损失权重

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

# ──────────────────────────────────────────────────────────────
# 特征列名定义
# ──────────────────────────────────────────────────────────────
CUTTING_PARAM_COLS = [
    "spindle_speed", "feed_per_tooth", "axial_depth", "radial_depth",
    "tooth_pass_freq", "spindle_freq", "feed_rate",
]
VIB_AXES  = ["vib_x", "vib_y", "vib_z"]
VIB_STATS = ["rms", "peak", "pp", "kurtosis", "skewness", "std", "fft_energy"]
VIB_FEAT_COLS = [f"{ax}_{st}" for ax in VIB_AXES for st in VIB_STATS]
INPUT_COLS    = CUTTING_PARAM_COLS + VIB_FEAT_COLS
RA_COL        = "Ra_um"
WAVINESS_COLS = [f"waviness_band_{i}" for i in range(N_FREQ_BINS)]


# ╔══════════════════════════════════════════════════════════════╗
# ║              模块 1: 数据集生成                              ║
# ╚══════════════════════════════════════════════════════════════╝

PARAM_RANGES = {
    "spindle_speed":  (2000, 8000),   # 转速 n (rpm)
    "feed_per_tooth": (0.02, 0.15),   # 每齿进给量 fz (mm/tooth)
    "axial_depth":    (0.2,  2.0),    # 轴向切深 ap (mm)
    "radial_depth":   (0.5,  5.0),    # 径向切深 ae (mm)
}


def sample_cutting_params(n: int) -> pd.DataFrame:
    """在参数空间内均匀随机采样 n 组切削参数并计算派生量"""
    data = {k: np.random.uniform(lo, hi, n) for k, (lo, hi) in PARAM_RANGES.items()}
    df = pd.DataFrame(data)
    df["tooth_pass_freq"] = df["spindle_speed"] / 60 * NUM_FLUTES
    df["spindle_freq"]    = df["spindle_speed"] / 60
    df["feed_rate"]       = df["feed_per_tooth"] * NUM_FLUTES * df["spindle_speed"]
    return df


def generate_vibration_signal(row: pd.Series) -> dict:
    """
    物理启发式振动信号生成器：
      振动 = Σ 切削谐波 + 主轴不平衡 + 固有频率共振 + 噪声
    返回 {'vib_x': array, 'vib_y': array, 'vib_z': array}
    """
    t     = np.linspace(0, SIGNAL_DURATION, int(FS * SIGNAL_DURATION), endpoint=False)
    ftpf  = row["tooth_pass_freq"]
    fsf   = row["spindle_freq"]
    f_amp = 0.5 * row["feed_per_tooth"] * row["axial_depth"] * row["radial_depth"]

    natural_freqs = {"vib_x": 420, "vib_y": 380, "vib_z": 510}
    result = {}
    for axis, fn in natural_freqs.items():
        s = np.zeros_like(t)
        # 齿通频率及前4阶倍频
        for k in range(1, 5):
            phi = np.random.uniform(0, 2 * np.pi)
            s  += (f_amp / k * (1 + 0.3 * np.random.randn())) * np.sin(2*np.pi*k*ftpf*t + phi)
        # 主轴不平衡分量
        s += 0.1 * f_amp * np.sin(2*np.pi*fsf*t + np.random.uniform(0, 2*np.pi))
        # 固有频率共振（衰减正弦）
        zeta, omega_n = 0.05, 2*np.pi*fn
        s += (f_amp * 0.3 / (2*zeta)) * np.exp(-zeta*omega_n*t) * \
             np.sin(omega_n * np.sqrt(1-zeta**2) * t)
        # 高斯噪声
        s += 0.05 * f_amp * np.random.randn(len(t))
        result[axis] = s
    return result


def compute_Ra(row: pd.Series, vib: dict) -> float:
    """
    经验公式计算表面粗糙度 Ra (μm)
    铣削 Ra 参考范围: 0.1 ~ 3.2 μm (精铣到半精铣)
    公式：Ra = K * fz^a * ap^b * ae^c / n^d + vibration_effect
    """
    fz, ap, ae, n = row["feed_per_tooth"], row["axial_depth"], row["radial_depth"], row["spindle_speed"]
    # 调整系数使 Ra 分布在 0.1~3.0 μm 范围内
    # fz 范围 0.02-0.15, ap 0.2-2.0, ae 0.5-5.0, n 2000-8000
    Ra_geom = 50.0 * (fz ** 1.5)                          # 进给量主导因素 (μm)
    Ra_cut  = Ra_geom * (ap ** 0.3) * (ae ** 0.15) * (4000.0 / n) ** 0.4
    vib_rms = np.sqrt(sum(np.mean(v**2) for v in vib.values()))
    Ra_vib  = 0.3 * vib_rms                               # 振动附加粗糙度
    Ra      = (Ra_cut + Ra_vib) * (1 + 0.08 * np.random.randn())
    return float(np.clip(Ra, 0.05, 5.0))


def compute_surface_waviness(row: pd.Series, vib: dict) -> np.ndarray:
    """
    从 Z 向振动频谱中提取 N_FREQ_BINS 个频带能量比
    （模拟表面起伏的频域分布）
    """
    z_sig   = vib["vib_z"]
    freqs   = np.fft.rfftfreq(len(z_sig), d=1.0/FS)
    fft_mag = np.abs(np.fft.rfft(z_sig)) / len(z_sig)
    edges   = np.linspace(0, FS/2, N_FREQ_BINS+1)
    energy  = np.array([
        np.mean(fft_mag[(freqs>=edges[i]) & (freqs<edges[i+1])])
        if np.any((freqs>=edges[i]) & (freqs<edges[i+1])) else 0.0
        for i in range(N_FREQ_BINS)
    ])
    energy += 0.01 * np.abs(np.random.randn(N_FREQ_BINS))
    energy  = np.clip(energy, 0, None)
    return energy / (energy.sum() + 1e-12)


def extract_vibration_features(vib: dict) -> dict:
    """从三向振动信号中提取时域+频域统计特征"""
    feats = {}
    for axis, sig in vib.items():
        feats[f"{axis}_rms"]      = np.sqrt(np.mean(sig**2))
        feats[f"{axis}_peak"]     = np.max(np.abs(sig))
        feats[f"{axis}_pp"]       = np.ptp(sig)
        feats[f"{axis}_kurtosis"] = float(pd.Series(sig).kurt())
        feats[f"{axis}_skewness"] = float(pd.Series(sig).skew())
        feats[f"{axis}_std"]      = np.std(sig)
        fft_mag = np.abs(np.fft.rfft(sig)) / len(sig)
        feats[f"{axis}_fft_energy"] = float(np.sum(fft_mag**2) / (np.sum(fft_mag**2)+1e-12))
    return feats


def generate_dataset(n_samples: int = 600) -> pd.DataFrame:
    """完整数据集生成流程"""
    print(f"[DataGen] 生成 {n_samples} 条铣削仿真记录 ...")
    params_df = sample_cutting_params(n_samples)
    Ra_list, wav_list, feat_list = [], [], []

    for _, row in params_df.iterrows():
        vib = generate_vibration_signal(row)
        Ra_list.append(compute_Ra(row, vib))
        wav_list.append(compute_surface_waviness(row, vib))
        feat_list.append(extract_vibration_features(vib))

    df = pd.concat([
        params_df.reset_index(drop=True),
        pd.DataFrame(feat_list),
        pd.Series(Ra_list, name=RA_COL),
        pd.DataFrame(wav_list, columns=WAVINESS_COLS),
    ], axis=1)

    print(f"[DataGen] 数据集形状: {df.shape} | "
          f"Ra ∈ [{df[RA_COL].min():.3f}, {df[RA_COL].max():.3f}] μm")
    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║              模块 2: 多任务神经网络                          ║
# ╚══════════════════════════════════════════════════════════════╝

class MillingDataset(Dataset):
    def __init__(self, X, y_ra, y_wav):
        self.X     = torch.tensor(X,     dtype=torch.float32)
        self.y_ra  = torch.tensor(y_ra,  dtype=torch.float32).unsqueeze(1)
        self.y_wav = torch.tensor(y_wav, dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y_ra[i], self.y_wav[i]


class MultiTaskMillingNet(nn.Module):
    """
    ┌─────────────────────────────────────────┐
    │  输入层 (n_features)                     │
    │      ↓                                   │
    │  共享编码器                               │
    │  FC(256)→BN→ReLU→Drop                   │
    │  FC(128)→BN→ReLU→Drop                   │
    │  FC(64) →BN→ReLU                        │
    │      ↙               ↘                  │
    │  Ra 预测头         Waviness 预测头        │
    │  FC(32)→ReLU       FC(32)→ReLU           │
    │  FC(1) →Softplus   FC(8)→Softmax         │
    └─────────────────────────────────────────┘
    """
    def __init__(self, n_features: int, n_bins: int = N_FREQ_BINS, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.ra_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),  nn.Softplus()   # 保证 Ra > 0
        )
        self.wav_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_bins), nn.Softmax(dim=1)  # 归一化概率分布
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.ra_head(z), self.wav_head(z)


# ╔══════════════════════════════════════════════════════════════╗
# ║              模块 3: 训练 & 预测框架                         ║
# ╚══════════════════════════════════════════════════════════════╝

class MillingPredictor:
    """铣削表面质量预测器（训练 + 推理 + 评估）"""

    def __init__(self):
        self.model      = None
        self.scaler_X   = StandardScaler()
        self.scaler_ra  = MinMaxScaler()
        self.history    = {"train_loss":[], "val_loss":[], "ra_loss":[], "wav_loss":[]}

    # ─── 数据准备 ───────────────────────────────
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> int:
        X      = df[INPUT_COLS].values
        y_ra   = df[RA_COL].values.reshape(-1, 1)
        y_wav  = df[WAVINESS_COLS].values

        X_tr, X_te, yra_tr, yra_te, yw_tr, yw_te = train_test_split(
            X, y_ra, y_wav, test_size=test_size, random_state=42)

        X_tr_s   = self.scaler_X.fit_transform(X_tr)
        X_te_s   = self.scaler_X.transform(X_te)
        yra_tr_s = self.scaler_ra.fit_transform(yra_tr).squeeze()
        yra_te_s = self.scaler_ra.transform(yra_te).squeeze()

        self.train_loader = DataLoader(MillingDataset(X_tr_s, yra_tr_s, yw_tr),
                                       batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader   = DataLoader(MillingDataset(X_te_s, yra_te_s, yw_te),
                                       batch_size=BATCH_SIZE, shuffle=False)
        self.X_te_s = X_te_s
        self.yra_te = yra_te.squeeze()
        self.yw_te  = yw_te
        print(f"[Data] 训练: {len(X_tr)} | 测试: {len(X_te)} | 特征维度: {X_tr_s.shape[1]}")
        return X_tr_s.shape[1]

    # ─── 构建模型 ───────────────────────────────
    def build_model(self, n_features: int):
        self.model     = MultiTaskMillingNet(n_features).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer, T_max=EPOCHS, eta_min=1e-5)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] 参数量: {n_params:,} | 设备: {DEVICE}")

    # ─── 损失函数 ────────────────────────────────
    @staticmethod
    def loss_fn(pred_ra, true_ra, pred_wav, true_wav):
        l_ra  = nn.functional.mse_loss(pred_ra.squeeze(), true_ra.squeeze())
        eps   = 1e-8
        l_wav = nn.functional.kl_div(
            (pred_wav + eps).log(), true_wav + eps, reduction="batchmean")
        return WEIGHT_RA * l_ra + WEIGHT_WAVINESS * l_wav, l_ra, l_wav

    # ─── 单 epoch 训练 ──────────────────────────
    def _train_epoch(self):
        self.model.train()
        tl = rl = wl = 0.0
        for X, y_ra, y_wav in self.train_loader:
            X, y_ra, y_wav = X.to(DEVICE), y_ra.to(DEVICE), y_wav.to(DEVICE)
            self.optimizer.zero_grad()
            pr, pw = self.model(X)
            loss, lr_, lw = self.loss_fn(pr, y_ra, pw, y_wav)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            tl += loss.item(); rl += lr_.item(); wl += lw.item()
        n = len(self.train_loader)
        return tl/n, rl/n, wl/n

    # ─── 验证 ────────────────────────────────────
    def _val_epoch(self):
        self.model.eval(); tl = 0.0
        with torch.no_grad():
            for X, y_ra, y_wav in self.val_loader:
                X, y_ra, y_wav = X.to(DEVICE), y_ra.to(DEVICE), y_wav.to(DEVICE)
                pr, pw = self.model(X)
                loss, _, _ = self.loss_fn(pr, y_ra, pw, y_wav)
                tl += loss.item()
        return tl / len(self.val_loader)

    # ─── 完整训练 ────────────────────────────────
    def train(self, epochs: int = EPOCHS):
        print(f"\n[Train] 开始训练，共 {epochs} epochs")
        best_val, best_state = float("inf"), None
        for ep in range(1, epochs+1):
            tl, rl, wl = self._train_epoch()
            vl = self._val_epoch()
            self.scheduler.step()
            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            self.history["ra_loss"].append(rl)
            self.history["wav_loss"].append(wl)
            if vl < best_val:
                best_val   = vl
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            if ep % 20 == 0 or ep == 1:
                print(f"  Epoch {ep:3d}/{epochs} | Train={tl:.4f} "
                      f"(Ra={rl:.4f} Wav={wl:.4f}) | Val={vl:.4f}")
        self.model.load_state_dict(best_state)
        print(f"[Train] 完成！最优验证损失: {best_val:.4f}")

    # ─── 评估 ────────────────────────────────────
    def evaluate(self) -> dict:
        self.model.eval()
        X_t = torch.tensor(self.X_te_s, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred_ra_norm, pred_wav = self.model(X_t)
        pred_ra = self.scaler_ra.inverse_transform(
                      pred_ra_norm.cpu().numpy()).squeeze()
        pred_wav = pred_wav.cpu().numpy()

        metrics = {
            "Ra_MAE" :  mean_absolute_error(self.yra_te, pred_ra),
            "Ra_RMSE":  np.sqrt(mean_squared_error(self.yra_te, pred_ra)),
            "Ra_R2"  :  r2_score(self.yra_te, pred_ra),
            "Wav_MAE":  mean_absolute_error(self.yw_te, pred_wav),
            "Wav_R2" :  r2_score(self.yw_te, pred_wav),
        }
        print("\n" + "="*50)
        print("📊 模型评估结果")
        print("="*50)
        print(f"  表面粗糙度 Ra:")
        print(f"    MAE  = {metrics['Ra_MAE']:.4f} μm")
        print(f"    RMSE = {metrics['Ra_RMSE']:.4f} μm")
        print(f"    R²   = {metrics['Ra_R2']:.4f}")
        print(f"  表面起伏频域分布:")
        print(f"    MAE  = {metrics['Wav_MAE']:.4f}")
        print(f"    R²   = {metrics['Wav_R2']:.4f}")
        print("="*50)

        self._pred_ra  = pred_ra
        self._pred_wav = pred_wav
        return metrics

    # ─── 推理接口（单样本）──────────────────────
    def predict(self, cutting_params: dict, vib_features: dict) -> dict:
        """
        输入切削参数和振动特征，返回预测结果

        Parameters
        ----------
        cutting_params : dict
            必填键: spindle_speed, feed_per_tooth, axial_depth, radial_depth
        vib_features : dict
            由 extract_vibration_features() 返回的振动统计特征字典

        Returns
        -------
        dict: {"Ra_um": float, "waviness_dist": list[float]}
        """
        # 补全派生量
        n  = cutting_params["spindle_speed"]
        fz = cutting_params["feed_per_tooth"]
        cutting_params.update({
            "tooth_pass_freq": n / 60 * NUM_FLUTES,
            "spindle_freq":    n / 60,
            "feed_rate":       fz * NUM_FLUTES * n,
        })
        row_dict = {**cutting_params, **vib_features}
        X_row    = np.array([[row_dict[c] for c in INPUT_COLS]])
        X_row_s  = self.scaler_X.transform(X_row)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_row_s, dtype=torch.float32).to(DEVICE)
            pr, pw = self.model(X_t)
        ra_val = float(self.scaler_ra.inverse_transform(
                           pr.cpu().numpy()).squeeze())
        return {
            "Ra_um":         ra_val,
            "waviness_dist": pw.cpu().numpy().squeeze().tolist()
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║              模块 4: 可视化                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def plot_training_history(history: dict):
    epochs = range(1, len(history["train_loss"])+1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train", lw=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   lw=2, ls="--")
    axes[0].set(title="Total Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["ra_loss"],  label="Ra Loss",       lw=2, color="steelblue")
    axes[1].plot(epochs, history["wav_loss"], label="Waviness Loss", lw=2, color="tomato")
    axes[1].set(title="Task-wise Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "01_training_loss.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_ra_prediction(true_ra, pred_ra):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmin, vmax = min(true_ra.min(), pred_ra.min()), max(true_ra.max(), pred_ra.max())
    axes[0].scatter(true_ra, pred_ra, alpha=0.6, s=40, color="steelblue", edgecolors="none")
    axes[0].plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5, label="Ideal")
    axes[0].set(title="Ra: Predicted vs True", xlabel="True Ra (μm)", ylabel="Pred Ra (μm)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    err = pred_ra - true_ra
    axes[1].hist(err, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", ls="--", lw=1.5)
    axes[1].axvline(err.mean(), color="orange", lw=1.5, label=f"Mean={err.mean():.4f}")
    axes[1].set(title="Ra Error Distribution", xlabel="Error (μm)", ylabel="Count")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "02_ra_prediction.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] {p}")


def plot_waviness_spectrum(true_wav, pred_wav, n_examples: int = 6):
    n_bins = true_wav.shape[1]
    indices = np.linspace(0, len(true_wav)-1, n_examples, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    x, w = np.arange(n_bins), 0.38
    for i, idx in enumerate(indices):
        axes[i].bar(x-w/2, true_wav[idx], w, label="True",      color="steelblue", alpha=0.8)
        axes[i].bar(x+w/2, pred_wav[idx], w, label="Predicted",  color="tomato",    alpha=0.8)
        axes[i].set(title=f"Sample #{idx}", xlabel="Freq Band",
                    ylabel="Normalized Energy", ylim=(0,1))
        axes[i].set_xticks(x); axes[i].set_xticklabels([f"B{j}" for j in range(n_bins)], fontsize=8)
        axes[i].legend(fontsize=7); axes[i].grid(axis="y", alpha=0.3)
    plt.suptitle("Surface Waviness Frequency Distribution: True vs Predicted", fontsize=12)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "03_waviness_spectrum.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot] {p}")


def plot_dashboard(history, true_ra, pred_ra, true_wav, pred_wav, metrics):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    epochs = range(1, len(history["train_loss"])+1)

    # ① 损失曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], label="Train", lw=1.5)
    ax1.plot(epochs, history["val_loss"],   label="Val",   lw=1.5, ls="--")
    ax1.set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    # ② Ra 散点
    ax2 = fig.add_subplot(gs[0, 1])
    vmin, vmax = min(true_ra.min(), pred_ra.min()), max(true_ra.max(), pred_ra.max())
    ax2.scatter(true_ra, pred_ra, alpha=0.5, s=20, color="steelblue")
    ax2.plot([vmin,vmax],[vmin,vmax],"r--",lw=1.5)
    ax2.set(title=f"Ra Prediction  R²={metrics['Ra_R2']:.3f}",
            xlabel="True Ra (μm)", ylabel="Pred Ra (μm)")
    ax2.grid(alpha=0.3)

    # ③ Ra 误差分布
    ax3 = fig.add_subplot(gs[0, 2])
    err = pred_ra - true_ra
    ax3.hist(err, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax3.axvline(0, color="red", ls="--")
    ax3.set(title=f"Ra Error  MAE={metrics['Ra_MAE']:.4f}μm",
            xlabel="Error (μm)", ylabel="Count")
    ax3.grid(alpha=0.3)

    # ④⑤⑥ 三样本频域对比
    n_bins = true_wav.shape[1]
    x, w   = np.arange(n_bins), 0.35
    for col_i, sid in enumerate([0, len(true_wav)//2, len(true_wav)-1]):
        ax = fig.add_subplot(gs[1, col_i])
        ax.bar(x-w/2, true_wav[sid], w, label="True",      color="steelblue", alpha=0.8)
        ax.bar(x+w/2, pred_wav[sid], w, label="Predicted",  color="tomato",    alpha=0.8)
        ax.set(title=f"Waviness Spectrum #{sid}",
               xlabel="Freq Band", ylabel="Norm. Energy")
        ax.set_xticks(x); ax.set_xticklabels([f"B{i}" for i in range(n_bins)], fontsize=7)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Milling Surface Quality Prediction Dashboard\n"
        f"Ra: MAE={metrics['Ra_MAE']:.4f}μm  R²={metrics['Ra_R2']:.3f} | "
        f"Waviness: MAE={metrics['Wav_MAE']:.4f}  R²={metrics['Wav_R2']:.3f}",
        fontsize=11
    )
    p = os.path.join(OUT_DIR, "00_dashboard.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot] 综合 Dashboard: {p}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        主程序                               ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    SEP = "=" * 62

    # ── Step 1: 生成数据集 ──────────────────────────────────
    print(SEP)
    print("  Step 1 | 生成模拟铣削数据集")
    print(SEP)
    df = generate_dataset(n_samples=600)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "milling_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DataGen] 数据集已保存: {csv_path}")

    # ── Step 2: 构建模型 ─────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 2 | 构建多任务预测模型")
    print(SEP)
    predictor  = MillingPredictor()
    n_features = predictor.prepare_data(df, test_size=0.2)
    predictor.build_model(n_features)

    # ── Step 3: 训练 ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 3 | 训练")
    print(SEP)
    predictor.train(epochs=EPOCHS)

    # ── Step 4: 评估 ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 4 | 评估")
    print(SEP)
    metrics = predictor.evaluate()

    # ── Step 5: 可视化 ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 5 | 可视化")
    print(SEP)
    plot_training_history(predictor.history)
    plot_ra_prediction(predictor.yra_te, predictor._pred_ra)
    plot_waviness_spectrum(predictor.yw_te, predictor._pred_wav)
    plot_dashboard(predictor.history,
                   predictor.yra_te, predictor._pred_ra,
                   predictor.yw_te,  predictor._pred_wav, metrics)

    # ── Step 6: 单样本预测演示 ──────────────────────────────
    print(f"\n{SEP}")
    print("  Step 6 | 单样本预测接口演示")
    print(SEP)
    demo_params = {
        "spindle_speed"  : 5000,   # rpm
        "feed_per_tooth" : 0.05,   # mm/tooth
        "axial_depth"    : 1.0,    # mm
        "radial_depth"   : 2.5,    # mm
    }
    demo_row = pd.Series({
        **demo_params,
        "tooth_pass_freq": demo_params["spindle_speed"]/60*NUM_FLUTES,
        "spindle_freq":    demo_params["spindle_speed"]/60,
        "feed_rate":       demo_params["feed_per_tooth"]*NUM_FLUTES*demo_params["spindle_speed"]
    })
    demo_vib      = generate_vibration_signal(demo_row)
    demo_vib_feat = extract_vibration_features(demo_vib)

    result = predictor.predict(demo_params, demo_vib_feat)

    print(f"\n  ▶ 切削参数")
    print(f"     转速       n  = {demo_params['spindle_speed']} rpm")
    print(f"     每齿进给  fz  = {demo_params['feed_per_tooth']} mm/tooth")
    print(f"     轴向切深  ap  = {demo_params['axial_depth']} mm")
    print(f"     径向切深  ae  = {demo_params['radial_depth']} mm")
    print(f"\n  ✅ 预测表面粗糙度 Ra = {result['Ra_um']:.4f} μm")
    print(f"  ✅ 预测表面起伏频域分布（各频带能量比）:")
    for i, v in enumerate(result["waviness_dist"]):
        bar = "█" * int(v * 40)
        print(f"     频带 {i}: {v:.4f}  {bar}")

    print(f"\n{SEP}")
    print(f"  ✅ 所有图表已保存到: {OUT_DIR}/")
    print(f"  ✅ 数据集已保存到:   {csv_path}")
    print(SEP)


if __name__ == "__main__":
    main()
