"""
铣削加工仿真数据生成
输入：切削参数(4) + 三向振动特征(12) = 16维
输出：表面粗糙度 Ra(1) + 表面起伏各阶谐波幅值(8维, μm)

频域设计：
  直接计算进给纹理的1~8阶谐波幅值，每阶幅值由Ra和振动决定
  物理意义：第k阶对应空间频率 k/fz (1/mm)
"""
import numpy as np
import csv
import os

np.random.seed(42)

N     = 500
Z     = 4
N_HARM = 8   # 谐波阶数

def vib_features(sig):
    return [float(np.mean(np.abs(sig))),
            float(np.std(sig)),
            float(np.sqrt(np.mean(sig**2))),
            float(np.max(np.abs(sig)))]

rows = []
for _ in range(N):
    n   = np.random.uniform(500,  3000)
    fz  = np.random.uniform(0.02, 0.15)
    ap  = np.random.uniform(0.5,  3.0)
    ae  = np.random.uniform(1.0,  8.0)
    f_z = n * Z / 60.0

    # 振动信号
    t = np.linspace(0, 0.1, 1000, endpoint=False)
    A = fz * ap * np.sqrt(ae)
    vib_x = sum(A/(k**1.5)*np.sin(2*np.pi*k*f_z*t+np.random.uniform(0,2*np.pi))
                for k in range(1,5)) + 0.002*A*np.random.randn(1000)
    vib_y = sum(0.7*A/(k**1.5)*np.sin(2*np.pi*k*f_z*t+np.random.uniform(0,2*np.pi))
                for k in range(1,5)) + 0.002*A*np.random.randn(1000)
    vib_z = sum(0.4*A/(k**1.5)*np.sin(2*np.pi*k*f_z*t+np.random.uniform(0,2*np.pi))
                for k in range(1,5)) + 0.001*A*np.random.randn(1000)

    fx, fy, fz_f = vib_features(vib_x), vib_features(vib_y), vib_features(vib_z)
    vib_rms = np.sqrt(fx[2]**2 + fy[2]**2 + fz_f[2]**2)

    # Ra (μm)
    Ra = 125.0*(fz**1.5)*(ap**0.3)*(ae**0.2)*(n**-0.3) + 2.0*vib_rms
    Ra *= (1 + 0.08*np.random.randn())
    Ra = max(Ra, 0.1)

    # 各阶谐波幅值 (μm)
    # 第k阶：空间频率 = k/fz (1/mm)
    # 幅值由Ra主导，振动增大高阶分量，转速影响高频衰减
    harm_amps = np.zeros(N_HARM)
    for k in range(N_HARM):
        order = k + 1
        # 基础幅值：Ra随阶次幂律衰减（铣削轮廓近似三角波）
        base = Ra / (order ** 1.5)
        # 振动贡献：对高阶分量影响更显著
        vib_effect = vib_rms * 0.4 * np.exp(-0.3 * order)
        # 转速影响：高转速时高阶分量相对增大（动态效应）
        speed_effect = 1.0 + 0.15 * (order / N_HARM) * (n / 3000.0)
        harm_amps[k] = (base + vib_effect) * speed_effect
        harm_amps[k] *= (1 + 0.06 * np.random.randn())
        harm_amps[k] = max(harm_amps[k], 0.0)

    row = ([round(n,2), round(fz,4), round(ap,3), round(ae,3)]
           + [round(v,6) for v in fx+fy+fz_f]
           + [round(Ra,4)]
           + [round(v,6) for v in harm_amps.tolist()])
    rows.append(row)

vib_cols  = [f'vib_{d}_{s}' for d in ['x','y','z']
             for s in ['mean_abs','std','rms','peak']]
harm_cols = [f'harm_{k+1}' for k in range(N_HARM)]
header    = ['n_rpm','fz_mm','ap_mm','ae_mm'] + vib_cols + ['Ra_um'] + harm_cols

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'milling_data.csv')
with open(out, 'w', newline='') as f:
    csv.writer(f).writerow(header)
    csv.writer(f).writerows(rows)

ra_vals   = [r[16] for r in rows]
harm_vals = np.array([r[17:25] for r in rows])
print(f"数据已生成：{out}，共 {N} 条，{len(header)} 列")
print(f"Ra    范围：{min(ra_vals):.3f} ~ {max(ra_vals):.3f} μm，均值 {np.mean(ra_vals):.3f} μm")
print(f"谐波幅值均值：{np.round(harm_vals.mean(0),4)}")
print(f"谐波幅值标准差：{np.round(harm_vals.std(0),4)}")
