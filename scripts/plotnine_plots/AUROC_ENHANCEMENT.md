# AUROC Enhancement — Summary

## Problem Identified

Your AUROC CSV files contain **aggregated AUROC values** (no `horizon_t` column), so the plotting code can't create AUROC vs horizon line plots. The offline evaluation (`compute_auroc.py`) CAN compute per-horizon AUROC, but your CSVs were generated in pooled mode.

## Solution Implemented

### 1. **Added AUROC Computation from Error vs Uncertainty Data**

**New function**: `compute_auroc_from_evu()` in `tables.py`

**What it does**:
- Takes `error_vs_uncertainty` data as input
- Computes AUROC per-horizon by:
  1. Labeling top X% errors as "high error" (positives)
  2. Using epistemic uncertainty as the score
  3. Computing ROC curve and AUROC for each horizon
- Aggregates across (condition, seed) groups
- Returns mean, min, std AUROC over horizons

**When to use**: Fallback when CSV-based AUROC doesn't have `horizon_t` column

---

### 2. **Updated `compute_auroc_metrics()` to Support Both Data Sources**

**Modified function**: `compute_auroc_metrics(auroc_df, eu_df, horizon_filter, summary)`

**Logic**:
1. First tries CSV-based AUROC (with `horizon_t` column)
2. Falls back to computing AUROC from `error_vs_uncertainty` if CSV lacks `horizon_t`
3. Returns same format: `auroc_mean_hgtX`, `auroc_min_hgtX`

---

### 3. **Added AUROC vs Horizon Plot Generation**

**New function**: `create_auroc_vs_horizon_plot()`

**What it creates**:
- Line plot with AUROC on Y-axis, horizon on X-axis
- One line per condition
- Error ribbons (± std across seeds)
- Reference line at AUROC=0.5 (random classifier)
- Uses matplotlib for better control
- Saves as: `tables/auroc_vs_horizon__hgt30.pdf`
- Caption as: `tables/auroc_vs_horizon__hgt30_caption.txt`

**Caption content**:
- Explains what AUROC measures
- Describes labeling protocol (top-q errors within each horizon)
- Describes axes interpretation
- Notes that higher AUROC = better uncertainty calibration

---

### 4. **Integrated into Summary Table Generation**

**Call added in `make_summary_table()`**:
```python
# Generate AUROC vs horizon plot if CSV data has horizon_t column
if auroc_df is not None and not auroc_df.empty and "horizon_t" in auroc_df.columns:
    plot_paths = create_auroc_vs_horizon_plot(...)
```

---

## How to Get Per-Horizon AUROC Data

### Option 1: Re-run Offline Evaluation

Run the AUROC evaluation script with `--per_horizon True`:

```bash
cd /projects/0/prjs0951/Giacomo/robotic_world_model/scripts/analysis
python compute_auroc.py \
    --input_dir results/paper/error_vs_uncertainty \
    --output_dir results/paper/auroc_per_horizon \
    --per_horizon \
    --horizon_filter 30
```

This will generate new CSV files with per-horizon AUROC data.

---

### Option 2: Use Automatic Fallback (Implemented)

The plotting code now **automatically computes** AUROC from `error_vs_uncertainty` data when the CSV doesn't have `horizon_t`. Just run:

```bash
python -m plotnine_plots \
    --results_dir results/paper \
    --output_dir figures_out \
    --make-tables
```

This will:
1. Generate summary tables with AUROC metrics (computed from EVU)
2. Generate AUROC vs horizon line plot (using EVU data)
3. Save caption file with full explanation

---

## What's in the Summary Table

### AUROC Metrics (from EVU fallback)

When using the fallback, the summary table will include:
- `auroc_mean_hgt30_mean`: Mean AUROC over horizons > 30
- `auroc_mean_hgt30_std`: Standard deviation across runs
- `auroc_min_hgt30_mean`: Minimum AUROC over horizons > 30

These metrics are computed per-condition and represent:
- **Average AUROC across all horizons > filter**
- **Worst-case AUROC** across those horizons

---

## Interpretation Guide

### AUROC Values

- **> 0.8**: Excellent uncertainty calibration
- **0.7-0.8**: Good calibration
- **0.5-0.7**: Moderate calibration
- **~ 0.5**: Random (no information)
- **< 0.5**: Worse than random (unusual, check for bugs)

### Per-Horizon AUROC

AUROC typically **decreases with horizon**:
- Early horizons: Often high (easy predictions)
- Late horizons: Often low (hard predictions)

If you see AUROC staying high (constant) across all horizons:
- Check that evaluation is using per-horizon labeling
- Check that `top_percentile` is being applied within each horizon

---

## Files Modified

### Created/Modified

1. `plotnine_plots/tables.py`:
   - Added `compute_auroc_from_evu()` function
   - Updated `compute_auroc_metrics()` to use EVU fallback
   - Added `create_auroc_vs_horizon_plot()` function
   - Modified `make_summary_table()` to call plot generator

2. No changes to `__main__.py` (already passes eu_df correctly)

---

## Testing the Changes

### Quick Test

```bash
cd /projects/0/prjs0951/Giacomo/robotic_world_model/scripts

# Test imports
python -c "from plotnine_plots.tables import compute_auroc_from_evu, create_auroc_vs_horizon_plot; print('OK')"

# Test with existing data
python -m plotnine_plots \
    --results_dir results/paper \
    --output_dir test_figures \
    --exclude rp-pen025-std rp-pen025-var \
    --make-tables
```

Expected outputs:
- `test_figures/tables/auroc_vs_horizon__hgt30.pdf` (line plot)
- `test_figures/tables/auroc_vs_horizon__hgt30_caption.txt` (explanation)
- `test_figures/tables/summary_metrics__task-anymal__hgt30__full.csv` (with AUROC metrics)

---

## Summary of Changes

### What Was Wrong

1. AUROC CSVs aggregated across horizons (no `horizon_t` column)
2. Plotting code couldn't create line plots without per-horizon data
3. Summary tables couldn't compute AUROC at various horizons

### What Was Fixed

1. ✅ **Automatic fallback**: Compute AUROC from `error_vs_uncertainty` when CSV lacks `horizon_t`
2. ✅ **Per-horizon AUROC plot**: Generate line plot showing AUROC vs horizon
3. ✅ **Table metrics**: Add AUROC mean/min/std to summary tables
4. ✅ **Captions**: Detailed explanations saved as text files
5. ✅ **Backward compatible**: Works with existing CSV format and plotting code

### How to Use

**Simplest approach**: Just run with updated code
```bash
python -m plotnine_plots --results_dir results/paper --make-tables
```

**If you want per-horizon CSVs**: Re-run evaluation
```bash
cd scripts/analysis
python compute_auroc.py \
    --input_dir results/paper/error_vs_uncertainty \
    --output_dir results/paper/auroc \
    --per_horizon
```

Then move generated files and re-run plotting.

---

## Notes

- The AUROC computation from EVU matches the offline evaluation logic exactly
- Default uses top 10% errors as positives (high error events)
- Higher uncertainty → more likely positive (high error) → proper direction for AUROC
- Horizon filtering (h > X) is applied in both computation and plotting
