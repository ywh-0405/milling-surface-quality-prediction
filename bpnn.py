"""
BP神经网络 - 铣削表面质量预测
任务1：预测表面粗糙度 Ra (μm)
任务2：预测表面起伏频域幅值 (8个分量, μm)
纯 numpy 实现，含 L2 正则化 + 早停
"""
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────
def relu(x):      return np.maximum(0, x)
def relu_d(x):    return (x > 0).astype(float)

def normalize(X, mean=None, std=None):
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-8
    return (X - mean) / std, mean, std

def denorm(X, mean, std):
    return X * std + mean

def mse(a, b):    return float(np.mean((a-b)**2))
def rmse(a, b):   return float(np.sqrt(np.mean((a-b)**2)))
def mae(a, b):    return float(np.mean(np.abs(a-b)))
def r2(a, b):
    ss_res = np.sum((a-b)**2)
    ss_tot = np.sum((a - a.mean())**2) + 1e-8
    return float(1 - ss_res/ss_tot)

# ─────────────────────────────────────────
# BPNN
# ─────────────────────────────────────────
class BPNN:
    def __init__(self, sizes, lr=0.005, momentum=0.9, l2=1e-4):
        self.lr, self.momentum, self.l2 = lr, momentum, l2
        self.W, self.b, self.vW, self.vb = [], [], [], []
        for i in range(len(sizes)-1):
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i])
            self.W.append(w);  self.b.append(np.zeros((1, sizes[i+1])))
            self.vW.append(np.zeros_like(w)); self.vb.append(np.zeros((1, sizes[i+1])))

    def forward(self, X):
        self.A = [X]
        a = X
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            a = relu(z) if i < len(self.W)-1 else z
            self.A.append(a)
        return a

    def backward(self, y):
        n = y.shape[0]
        d = (self.A[-1] - y) * 2 / n
        for i in reversed(range(len(self.W))):
            dw = self.A[i].T @ d + self.l2 * self.W[i]
            db = d.sum(0, keepdims=True)
            self.vW[i] = self.momentum*self.vW[i] - self.lr*dw
            self.vb[i] = self.momentum*self.vb[i] - self.lr*db
            self.W[i] += self.vW[i];  self.b[i] += self.vb[i]
            if i > 0:
                d = (d @ self.W[i].T) * relu_d(self.A[i])

    def fit(self, Xtr, ytr, Xval, yval, epochs=600, batch=32, patience=60):
        best_val, wait, best_W, best_b = np.inf, 0, None, None
        tr_losses, val_losses = [], []
        for ep in range(1, epochs+1):
            idx = np.random.permutation(len(Xtr))
            Xs, ys = Xtr[idx], ytr[idx]
            ep_loss = []
            for s in range(0, len(Xs), batch):
                xb, yb = Xs[s:s+batch], ys[s:s+batch]
                ep_loss.append(mse(self.forward(xb), yb))
                self.backward(yb)
            tl = float(np.mean(ep_loss))
            vl = mse(self.forward(Xval), yval)
            tr_losses.append(tl); val_losses.append(vl)
            if ep % 100 == 0:
                print(f"  Epoch {ep:4d} | train={tl:.6f} | val={vl:.6f}")
            if vl < best_val - 1e-6:
                best_val, wait = vl, 0
                best_W = [w.copy() for w in self.W]
                best_b = [b.copy() for b in self.b]
            else:
                wait += 1
                if wait >= patience:
                    print(f"  早停于 Epoch {ep}，最佳验证损失={best_val:.6f}")
                    break
        if best_W:
            self.W, self.b = best_W, best_b
        return tr_losses, val_losses

    def predict(self, X):
        return self.forward(X)

# ─────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────
with open(os.path.join(BASE, 'milling_data.csv'), newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    data = np.array([[float(v) for v in row] for row in reader])

ra_col    = header.index('Ra_um')
fft_start = next(i for i,h in enumerate(header) if h.startswith('harm_') or h.startswith('harmonic_') or h.startswith('fft_bin_'))

X_raw = data[:, :ra_col]           # 16维输入
y_Ra  = data[:, ra_col:ra_col+1]   # Ra
y_fft = data[:, fft_start:fft_start+8]  # 8维频域

# 划分 train/val/test = 70/10/20
np.random.seed(0)
idx   = np.random.permutation(len(X_raw))
n     = len(X_raw)
n_tr  = int(0.7*n); n_val = int(0.1*n)
tr_i, val_i, te_i = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]

X_n, Xm, Xs       = normalize(X_raw)
yRa_n, Rm, Rs     = normalize(y_Ra)
yFft_n, Fm, Fs_   = normalize(y_fft)

Xtr,  yRa_tr,  yFft_tr  = X_n[tr_i],  yRa_n[tr_i],  yFft_n[tr_i]
Xval, yRa_val, yFft_val  = X_n[val_i], yRa_n[val_i], yFft_n[val_i]
Xte,  yRa_te,  yFft_te   = X_n[te_i],  yRa_n[te_i],  yFft_n[te_i]

n_in = Xtr.shape[1]  # 16

# ─────────────────────────────────────────
# 训练
# ─────────────────────────────────────────
print("=" * 50)
print("网络1：预测表面粗糙度 Ra")
print("=" * 50)
net_Ra = BPNN([n_in, 32, 16, 1], lr=0.005, momentum=0.9, l2=1e-4)
loss_Ra_tr, loss_Ra_val = net_Ra.fit(Xtr, yRa_tr, Xval, yRa_val, epochs=600, patience=60)

print("\n" + "=" * 50)
print("网络2：预测表面起伏频域幅值")
print("=" * 50)
net_fft = BPNN([n_in, 64, 32, 8], lr=0.003, momentum=0.9, l2=1e-3)
loss_fft_tr, loss_fft_val = net_fft.fit(Xtr, yFft_tr, Xval, yFft_val, epochs=600, patience=60)

# ─────────────────────────────────────────
# 测试集评估
# ─────────────────────────────────────────
Ra_pred = denorm(net_Ra.predict(Xte),   Rm, Rs)
Ra_true = denorm(yRa_te,                Rm, Rs)
Ft_pred = denorm(net_fft.predict(Xte),  Fm, Fs_)
Ft_true = denorm(yFft_te,               Fm, Fs_)

print("\n" + "=" * 50)
print("测试集评估")
print("=" * 50)
print(f"  Ra  | RMSE={rmse(Ra_true,Ra_pred):.4f} μm | MAE={mae(Ra_true,Ra_pred):.4f} μm | R²={r2(Ra_true.flatten(),Ra_pred.flatten()):.4f}")
print(f"  FFT | RMSE={rmse(Ft_true,Ft_pred):.4f} μm | MAE={mae(Ft_true,Ft_pred):.4f} μm | R²={r2(Ft_true.flatten(),Ft_pred.flatten()):.4f}")

# ─────────────────────────────────────────
# 写结果文件
# ─────────────────────────────────────────
# 1. 训练损失
max_ep = max(len(loss_Ra_tr), len(loss_fft_tr))
with open(os.path.join(BASE, 'training_loss.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['epoch','Ra_train','Ra_val','FFT_train','FFT_val'])
    for i in range(max_ep):
        w.writerow([i+1,
                    round(loss_Ra_tr[i],8)  if i < len(loss_Ra_tr)  else '',
                    round(loss_Ra_val[i],8) if i < len(loss_Ra_val) else '',
                    round(loss_fft_tr[i],8) if i < len(loss_fft_tr) else '',
                    round(loss_fft_val[i],8)if i < len(loss_fft_val)else ''])

# 2. Ra 预测结果
with open(os.path.join(BASE, 'result_Ra.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Ra_true_um','Ra_pred_um','abs_error_um'])
    for t, p in zip(Ra_true.flatten(), Ra_pred.flatten()):
        w.writerow([round(t,4), round(p,4), round(abs(t-p),4)])

# 3. FFT 预测结果
with open(os.path.join(BASE, 'result_fft.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([f'true_bin{i}' for i in range(8)] + [f'pred_bin{i}' for i in range(8)])
    for t, p in zip(Ft_true, Ft_pred):
        w.writerow([round(v,6) for v in t] + [round(v,6) for v in p])

# 4. 汇总
with open(os.path.join(BASE, 'summary.txt'), 'w') as f:
    f.write("铣削表面质量 BPNN 预测结果汇总\n")
    f.write("=" * 45 + "\n")
    f.write(f"样本总数：{n}  训练：{n_tr}  验证：{n_val}  测试：{n-n_tr-n_val}\n")
    f.write(f"输入维度：{n_in}（切削参数4 + 三向振动特征12）\n\n")
    f.write("网络1（Ra）：16→32→16→1，ReLU，L2=1e-4\n")
    f.write(f"  RMSE = {rmse(Ra_true,Ra_pred):.4f} μm\n")
    f.write(f"  MAE  = {mae(Ra_true,Ra_pred):.4f} μm\n")
    f.write(f"  R²   = {r2(Ra_true.flatten(),Ra_pred.flatten()):.4f}\n\n")
    f.write("网络2（FFT）：16→64→32→8，ReLU，L2=1e-3\n")
    f.write(f"  RMSE = {rmse(Ft_true,Ft_pred):.4f} μm\n")
    f.write(f"  MAE  = {mae(Ft_true,Ft_pred):.4f} μm\n")
    f.write(f"  R²   = {r2(Ft_true.flatten(),Ft_pred.flatten()):.4f}\n")

print("\n结果已写入：training_loss.csv / result_Ra.csv / result_fft.csv / summary.txt")
