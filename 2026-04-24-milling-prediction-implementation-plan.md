# Milling Surface Quality Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the current milling prediction prototype into a dual-mode project with a useful root README, a lightweight training path, a research training path, and future-ready real-data interfaces.

**Architecture:** Keep the existing prototype scripts as legacy references, add a new `src/` package with shared schema/data/feature code, and expose separate `lite` and `research` training entry points over the same processed dataset contract. Put the main explanatory burden into a detailed root `README.md` written for a mechanical-manufacturing teacher rather than a software engineer.

**Tech Stack:** Python, `numpy`, `pandas`, `scikit-learn`, optional `matplotlib`, optional `torch`, Git

---

### Task 1: Establish the new project skeleton and preserve legacy scripts

**Files:**
- Create: `/home/y/Desktop/BPNN/src/__init__.py`
- Create: `/home/y/Desktop/BPNN/data/raw/.gitkeep`
- Create: `/home/y/Desktop/BPNN/data/processed/.gitkeep`
- Create: `/home/y/Desktop/BPNN/data/synthetic/.gitkeep`
- Create: `/home/y/Desktop/BPNN/outputs/lite/.gitkeep`
- Create: `/home/y/Desktop/BPNN/outputs/research/.gitkeep`
- Create: `/home/y/Desktop/BPNN/examples/.gitkeep`
- Create: `/home/y/Desktop/BPNN/legacy/.gitkeep`
- Modify: `/home/y/Desktop/BPNN/bpnn.py`
- Modify: `/home/y/Desktop/BPNN/generate_data.py`
- Modify: `/home/y/Desktop/BPNN/milling_prediction_framework.py`

- [ ] **Step 1: Write the failing structural test**

```python
from pathlib import Path

def test_project_directories_exist():
    root = Path("/home/y/Desktop/BPNN")
    assert (root / "src").is_dir()
    assert (root / "data" / "raw").is_dir()
    assert (root / "data" / "processed").is_dir()
    assert (root / "data" / "synthetic").is_dir()
    assert (root / "outputs" / "lite").is_dir()
    assert (root / "outputs" / "research").is_dir()
    assert (root / "examples").is_dir()
    assert (root / "legacy").is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout.py::test_project_directories_exist -v`
Expected: FAIL because the new directories do not exist yet.

- [ ] **Step 3: Create the directory skeleton and prepare legacy preservation**

```python
from pathlib import Path

root = Path("/home/y/Desktop/BPNN")
for rel in [
    "src",
    "data/raw",
    "data/processed",
    "data/synthetic",
    "outputs/lite",
    "outputs/research",
    "examples",
    "legacy",
]:
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        keep = path / ".gitkeep"
        keep.touch(exist_ok=True)
```
Move the three prototype scripts into `legacy/`, then add small wrapper scripts at the root that import or document the new location so existing references do not silently break.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_layout.py::test_project_directories_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src data outputs examples legacy bpnn.py generate_data.py milling_prediction_framework.py tests/test_layout.py
git commit -m "refactor: add dual-mode project skeleton"
```

### Task 2: Define shared schema, field names, and processed-data contract

**Files:**
- Create: `/home/y/Desktop/BPNN/src/config.py`
- Create: `/home/y/Desktop/BPNN/src/schemas.py`
- Test: `/home/y/Desktop/BPNN/tests/test_schemas.py`

- [ ] **Step 1: Write the failing schema test**

```python
from src.schemas import REQUIRED_INPUT_COLUMNS, TARGET_COLUMNS, derive_process_columns

def test_schema_lists_are_stable():
    assert "spindle_speed_rpm" in REQUIRED_INPUT_COLUMNS
    assert "tool_flutes" in REQUIRED_INPUT_COLUMNS
    assert TARGET_COLUMNS == ["Ra_um"] + [f"freq_bin_{i}" for i in range(1, 9)]

def test_derive_process_columns():
    row = {
        "spindle_speed_rpm": 6000.0,
        "feed_per_tooth_mm": 0.05,
        "tool_flutes": 4,
    }
    derived = derive_process_columns(row)
    assert round(derived["spindle_freq_hz"], 4) == 100.0
    assert round(derived["tooth_pass_freq_hz"], 4) == 400.0
    assert round(derived["feed_rate_mm_min"], 4) == 1200.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL with import or attribute errors because `schemas.py` does not exist yet.

- [ ] **Step 3: Write the minimal schema implementation**

```python
REQUIRED_BASE_COLUMNS = [
    "sample_id",
    "spindle_speed_rpm",
    "feed_per_tooth_mm",
    "axial_depth_mm",
    "radial_depth_mm",
    "tool_flutes",
]

REQUIRED_VIBRATION_COLUMNS = [
    "vib_x_rms", "vib_x_std", "vib_x_peak", "vib_x_pp",
    "vib_y_rms", "vib_y_std", "vib_y_peak", "vib_y_pp",
    "vib_z_rms", "vib_z_std", "vib_z_peak", "vib_z_pp",
]

REQUIRED_INPUT_COLUMNS = REQUIRED_BASE_COLUMNS + REQUIRED_VIBRATION_COLUMNS
DERIVED_COLUMNS = ["spindle_freq_hz", "tooth_pass_freq_hz", "feed_rate_mm_min"]
TARGET_COLUMNS = ["Ra_um"] + [f"freq_bin_{i}" for i in range(1, 9)]

def derive_process_columns(row):
    spindle = float(row["spindle_speed_rpm"])
    fz = float(row["feed_per_tooth_mm"])
    flutes = float(row["tool_flutes"])
    return {
        "spindle_freq_hz": spindle / 60.0,
        "tooth_pass_freq_hz": spindle * flutes / 60.0,
        "feed_rate_mm_min": spindle * flutes * fz,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/schemas.py tests/test_schemas.py
git commit -m "feat: add shared data schema contract"
```

### Task 3: Build data preparation for synthetic and future real data

**Files:**
- Create: `/home/y/Desktop/BPNN/src/data_io.py`
- Create: `/home/y/Desktop/BPNN/examples/sample_real_data_template.csv`
- Create: `/home/y/Desktop/BPNN/examples/sample_input.csv`
- Test: `/home/y/Desktop/BPNN/tests/test_data_io.py`

- [ ] **Step 1: Write the failing data preparation test**

```python
from pathlib import Path
import pandas as pd
from src.data_io import prepare_summary_table

def test_prepare_summary_table_writes_features_and_targets(tmp_path):
    df = pd.DataFrame([{
        "sample_id": "S001",
        "spindle_speed_rpm": 6000,
        "feed_per_tooth_mm": 0.05,
        "axial_depth_mm": 1.0,
        "radial_depth_mm": 2.0,
        "tool_flutes": 4,
        "vib_x_rms": 0.10, "vib_x_std": 0.09, "vib_x_peak": 0.22, "vib_x_pp": 0.41,
        "vib_y_rms": 0.08, "vib_y_std": 0.07, "vib_y_peak": 0.19, "vib_y_pp": 0.36,
        "vib_z_rms": 0.06, "vib_z_std": 0.05, "vib_z_peak": 0.15, "vib_z_pp": 0.30,
        "Ra_um": 0.85,
        "freq_bin_1": 0.21, "freq_bin_2": 0.18, "freq_bin_3": 0.15, "freq_bin_4": 0.13,
        "freq_bin_5": 0.11, "freq_bin_6": 0.09, "freq_bin_7": 0.08, "freq_bin_8": 0.05,
    }])
    src = tmp_path / "input.csv"
    out_dir = tmp_path / "processed"
    df.to_csv(src, index=False)

    result = prepare_summary_table(src, out_dir)

    assert (out_dir / "features.csv").exists()
    assert (out_dir / "targets.csv").exists()
    assert "spindle_freq_hz" in result["features"].columns
    assert list(result["targets"].columns) == ["sample_id", "Ra_um"] + [f"freq_bin_{i}" for i in range(1, 9)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_io.py -v`
Expected: FAIL because `data_io.py` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
import pandas as pd
from pathlib import Path
from src.schemas import REQUIRED_INPUT_COLUMNS, TARGET_COLUMNS, derive_process_columns

def prepare_summary_table(input_csv, output_dir):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    missing = [c for c in REQUIRED_INPUT_COLUMNS + TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    derived = df.apply(lambda row: pd.Series(derive_process_columns(row)), axis=1)
    features = pd.concat([df[["sample_id"] + REQUIRED_INPUT_COLUMNS], derived], axis=1)
    targets = df[["sample_id"] + TARGET_COLUMNS].copy()

    features.to_csv(output_dir / "features.csv", index=False)
    targets.to_csv(output_dir / "targets.csv", index=False)
    return {"features": features, "targets": targets}
```
Also create `examples/sample_real_data_template.csv` and `examples/sample_input.csv` using the same field names.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_io.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_io.py examples/sample_real_data_template.csv examples/sample_input.csv tests/test_data_io.py
git commit -m "feat: add processed-data preparation flow"
```

### Task 4: Implement the lite training and prediction path

**Files:**
- Create: `/home/y/Desktop/BPNN/src/models_lite.py`
- Create: `/home/y/Desktop/BPNN/src/train_lite.py`
- Create: `/home/y/Desktop/BPNN/src/predict.py`
- Test: `/home/y/Desktop/BPNN/tests/test_lite_pipeline.py`

- [ ] **Step 1: Write the failing lite pipeline test**

```python
import pandas as pd
from src.models_lite import train_lite_models

def test_train_lite_models_returns_metrics():
    features = pd.DataFrame([
        {"sample_id": "S1", "spindle_speed_rpm": 4000, "feed_per_tooth_mm": 0.03, "axial_depth_mm": 0.8, "radial_depth_mm": 1.5,
         "tool_flutes": 4, "vib_x_rms": 0.12, "vib_x_std": 0.10, "vib_x_peak": 0.22, "vib_x_pp": 0.42,
         "vib_y_rms": 0.10, "vib_y_std": 0.09, "vib_y_peak": 0.19, "vib_y_pp": 0.38,
         "vib_z_rms": 0.07, "vib_z_std": 0.06, "vib_z_peak": 0.15, "vib_z_pp": 0.31,
         "spindle_freq_hz": 66.6667, "tooth_pass_freq_hz": 266.6667, "feed_rate_mm_min": 480.0},
    ] * 6)
    targets = pd.DataFrame([
        {"sample_id": "S1", "Ra_um": 0.70, "freq_bin_1": 0.20, "freq_bin_2": 0.18, "freq_bin_3": 0.15, "freq_bin_4": 0.13,
         "freq_bin_5": 0.11, "freq_bin_6": 0.09, "freq_bin_7": 0.08, "freq_bin_8": 0.06},
    ] * 6)

    result = train_lite_models(features, targets)

    assert "ra_model" in result
    assert "freq_model" in result
    assert "metrics" in result
    assert "ra_mae" in result["metrics"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lite_pipeline.py -v`
Expected: FAIL because the lite training modules do not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def train_lite_models(features, targets):
    merged = features.merge(targets, on="sample_id")
    X = merged.drop(columns=["sample_id", "Ra_um"] + [f"freq_bin_{i}" for i in range(1, 9)])
    y_ra = merged["Ra_um"]
    y_freq = merged[[f"freq_bin_{i}" for i in range(1, 9)]]

    ra_model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), random_state=42, max_iter=2000)),
    ])
    freq_model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(48, 24), random_state=42, max_iter=2000))),
    ])

    ra_model.fit(X, y_ra)
    freq_model.fit(X, y_freq)

    ra_pred = ra_model.predict(X)
    return {
        "ra_model": ra_model,
        "freq_model": freq_model,
        "metrics": {"ra_mae": float(mean_absolute_error(y_ra, ra_pred))},
        "feature_columns": list(X.columns),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lite_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models_lite.py src/train_lite.py src/predict.py tests/test_lite_pipeline.py
git commit -m "feat: add lite training and prediction path"
```

### Task 5: Implement the research training and evaluation path

**Files:**
- Create: `/home/y/Desktop/BPNN/src/models_research.py`
- Create: `/home/y/Desktop/BPNN/src/train_research.py`
- Create: `/home/y/Desktop/BPNN/src/evaluate.py`
- Test: `/home/y/Desktop/BPNN/tests/test_research_interfaces.py`

- [ ] **Step 1: Write the failing interface test**

```python
from src.models_research import build_research_model

def test_build_research_model_has_two_heads():
    model = build_research_model(n_features=20, n_freq_bins=8)
    assert hasattr(model, "ra_head")
    assert hasattr(model, "freq_head")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_research_interfaces.py -v`
Expected: FAIL because the research model module does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

def build_research_model(n_features, n_freq_bins):
    if nn is None:
        raise ImportError("PyTorch is required for research mode")

    class ResearchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.ra_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            self.freq_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_freq_bins))

        def forward(self, x):
            z = self.encoder(x)
            return self.ra_head(z), self.freq_head(z)

    return ResearchNet()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_research_interfaces.py -v`
Expected: PASS when `torch` is installed, or SKIP if the test marks PyTorch as optional.

- [ ] **Step 5: Commit**

```bash
git add src/models_research.py src/train_research.py src/evaluate.py tests/test_research_interfaces.py
git commit -m "feat: add research training interfaces"
```

### Task 6: Replace the root README with the teacher-facing project document

**Files:**
- Modify: `/home/y/Desktop/BPNN/README.md`
- Test: `/home/y/Desktop/BPNN/tests/test_readme_smoke.py`

- [ ] **Step 1: Write the failing README smoke test**

```python
from pathlib import Path

def test_root_readme_mentions_core_sections():
    content = Path("/home/y/Desktop/BPNN/README.md").read_text(encoding="utf-8")
    assert "项目背景" in content
    assert "输入与输出" in content
    assert "快速开始" in content
    assert "未来如何接入真实实验数据" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_readme_smoke.py -v`
Expected: FAIL because the current README only contains the repository title.

- [ ] **Step 3: Write the final README content**

```markdown
# milling-surface-quality-prediction

## 项目背景

本项目面向铣削加工表面质量预测。目标是在已知切削参数和加工振动信息的前提下，预测加工后的表面粗糙度 `Ra`，并进一步预测表面起伏在频域中的分布特征。

## 输入与输出

- 输入：主轴转速、每齿进给量、轴向切深、径向切深、刀具刃数、三向振动特征
- 输出：`Ra_um` 和 `freq_bin_1 ~ freq_bin_8`

## 快速开始

1. 准备 Python 环境
2. 安装轻量版或研究版依赖
3. 运行数据准备命令
4. 运行 `lite` 或 `research` 训练命令
5. 使用预测命令生成新工况结果

## 未来如何接入真实实验数据

项目支持汇总表模式和原始文件夹模式，最终都会整理成 `data/processed/features.csv` 和 `data/processed/targets.csv` 后再训练。
```
Expand that base into the full teacher-facing README required by the spec, including project purpose, current-vs-future explanation, directory table, command examples, output descriptions, and FAQ.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_readme_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_readme_smoke.py
git commit -m "docs: write teacher-facing project readme"
```

### Task 7: Verify the combined workflow on synthetic data

**Files:**
- Modify: `/home/y/Desktop/BPNN/src/synthetic_data.py`
- Modify: `/home/y/Desktop/BPNN/src/train_lite.py`
- Modify: `/home/y/Desktop/BPNN/src/train_research.py`
- Test: `/home/y/Desktop/BPNN/tests/test_end_to_end_smoke.py`

- [ ] **Step 1: Write the failing end-to-end smoke test**

```python
from pathlib import Path
from src.synthetic_data import generate_synthetic_dataset

def test_synthetic_dataset_can_be_generated(tmp_path):
    out = tmp_path / "synthetic.csv"
    df = generate_synthetic_dataset(20)
    assert len(df) == 20
    assert "Ra_um" in df.columns
    assert "freq_bin_8" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_end_to_end_smoke.py -v`
Expected: FAIL because `synthetic_data.py` is not implemented yet.

- [ ] **Step 3: Write the minimal implementation**

```python
import pandas as pd
from src.synthetic_data import _generate_single_sample

def generate_synthetic_dataset(n_samples):
    rows = [_generate_single_sample(i) for i in range(n_samples)]
    return pd.DataFrame(rows)
```
The private helper `_generate_single_sample()` should contain the extracted synthetic-generation logic migrated from the current prototype.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_end_to_end_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_data.py src/train_lite.py src/train_research.py tests/test_end_to_end_smoke.py
git commit -m "test: add synthetic end-to-end smoke coverage"
```

## Self-Review

- Spec coverage: the plan covers directory restructuring, data schema, processed-data generation, lite mode, research mode, teacher-facing README, and synthetic verification. The only deferred area is full profile-sequence modeling, which the spec explicitly excludes from the current training scope.
- Placeholder scan: no `TODO`, `TBD`, or unnamed placeholder dependencies remain.
- Type consistency: feature names, target names, and derived column names are consistent across Tasks 2 through 7 and match the design document.
