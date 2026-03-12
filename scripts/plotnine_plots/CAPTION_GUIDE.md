# Plot Subtitle Removal and Caption Generation

## Summary of Changes

Subtitles have been **removed from all plots** and instead **saved as separate caption text files** alongside PDFs. This makes plots cleaner and makes captions more flexible for paper writing.

---

## What Changed

### 1. `save_plot()` Function Modified

**File**: `plotnine_plots/styles.py`

**Change**: Added optional `caption` parameter to `save_plot()` function.

**New signature**:
```python
def save_plot(
    p: ggplot,
    output_dir: Path,
    name: str,
    width: float = 6,
    height: float = 3.8,
    dpi: int = 300,
    formats: Optional[list[str]] = None,
    caption: Optional[str] = None,  # NEW
) -> list[Path]:
```

**Behavior**:
- If `caption` is provided, saves it as `{name}_caption.txt` alongside the plot
- Caption file contains detailed explanation of the plot
- Returns list of all paths (plot + caption file if provided)

---

### 2. `plot_decision.py` Updated

**Plots updated**:
1. **Risk-Coverage Curve** (`_plot_risk_coverage`)
2. **AUROC vs Horizon** (`_plot_auroc_vs_horizon`)
3. **AUROC Summary** (`_plot_auroc_summary`)

**Changes**:
- Removed `subtitle=` parameter from `labs()` calls
- Built detailed `caption` string instead
- Passed `caption=caption` to `save_plot()` function

---

### 3. Risk-Coverage Caption

**File generated**: `decision_utility_risk_coverage{suffix}_caption.txt`

**Caption content**:
```
Risk-Coverage Curve

Risk = mean prediction error on retained subset
Filtering by lowest epistemic uncertainty (ranking-based)
Horizon filter: h > 30

X-axis: Coverage (fraction kept) - Coverage=1 means keep all samples (no filtering), coverage<1 means keep only lowest-uncertainty samples
Y-axis: Risk (mean prediction error) - Error measured on the retained subset
Vertical dashed line: coverage=0.8 (common operating point for selective prediction)
```

**Key points explained**:
- What risk represents
- How filtering is applied (ranking by uncertainty)
- Horizon filter used (if any)
- X-axis interpretation
- Y-axis interpretation
- Visual aids (dashed line at 0.8)

---

### 4. AUROC vs Horizon Caption

**File generated**: `decision_utility_auroc_vs_horizon{suffix}_caption.txt`

**Caption content**:
```
AUROC vs Prediction Horizon

AUROC computed per-horizon (positives = top-q errors within each horizon)
Horizontal dashed line: AUROC=0.5 (random classifier performance)
Horizon filter: h > 30

X-axis: Prediction horizon (steps)
Y-axis: AUROC - Area Under the Receiver Operating Characteristic curve
Higher AUROC indicates better uncertainty calibration (distinguishes high-error predictions)
```

**Key points explained**:
- How AUROC is computed (per-horizon)
- Labeling protocol (top-q errors per horizon)
- 0.5 reference line (random classifier)
- Horizon filter used (if any)
- Axes interpretation
- What AUROC measures (uncertainty calibration)

---

### 5. AUROC Summary Caption

**File generated**: `decision_utility_auroc_summary{suffix}_caption.txt`

**Caption content**:
```
AUROC Summary by Condition

Mean AUROC over horizons > 30

X-axis: Condition
Y-axis: Mean AUROC - Aggregated across filtered horizons
Error bars: Standard deviation across seeds
Horizontal dashed line: AUROC=0.5 (random classifier performance)
```

**Key points explained**:
- How mean AUROC is computed (over filtered horizons)
- X-axis (conditions)
- Y-axis (mean AUROC)
- Error bars (std across seeds)
- 0.5 reference line

---

## Why This Change?

### Advantages

1. **Cleaner Plots**
   - No subtitles cluttering the plot
   - Focuses visual attention on the data
   - Cleaner for publication

2. **More Flexible Captions**
   - Captions can be edited without regenerating plots
   - Captions can be used verbatim or modified for paper
   - Captions contain more detailed explanations than typical subtitles

3. **Better Paper Writing**
   - Can reference caption files directly
   - Can copy-paste caption text into paper
   - Easier to adapt caption for word count requirements

4. **Backward Compatible**
   - If no caption provided, works exactly as before
   - `caption=None` is default
   - All existing code still works

---

## File Naming Convention

### Caption Files
```
{plot_name}_caption.txt
```

### Examples
```
decision_utility_risk_coverage__hgt30_caption.txt
decision_utility_auroc_vs_horizon__hgt30_caption.txt
decision_utility_auroc_summary__hgt30_caption.txt
```

### Always Generated Together
For each plot, you get:
- `{plot_name}.pdf` - The visual plot
- `{plot_name}_caption.txt` - The detailed explanation

---

## Usage in Paper Writing

### 1. Reference Caption File
```
As shown in Figure X, the risk-coverage curves demonstrate how filtering
by epistemic uncertainty reduces prediction error. At 80% coverage,
the error is reduced by X% compared to no filtering (see caption in
decision_utility_risk_coverage__hgt30_caption.txt for details).
```

### 2. Copy Caption Text
You can directly use or adapt the caption text:
```bash
cat decision_utility_auroc_vs_horizon__hgt30_caption.txt
```

Output:
```
AUROC vs Prediction Horizon

AUROC computed per-horizon (positives = top-q errors within each horizon)
Horizontal dashed line: AUROC=0.5 (random classifier performance)
Horizon filter: h > 30

X-axis: Prediction horizon (steps)
Y-axis: AUROC - Area Under the Receiver Operating Characteristic curve
Higher AUROC indicates better uncertainty calibration (distinguishes high-error predictions)
```

### 3. Edit Caption
Edit the caption file as needed for your paper:
- Adjust wording
- Change "Figure X" reference
- Add additional notes
- Condense for word count

---

## Testing

### Verify Captions Are Generated

```bash
# Check for caption files
ls figures_out/decision_utility/*_caption.txt

# View a specific caption
cat figures_out/decision_utility/decision_utility_risk_coverage__hgt30_caption.txt
```

### Verify AUROC Line Plot Exists

```bash
# Check that line plot is generated (not just bar)
ls figures_out/decision_utility/*.pdf

# Should see:
# - decision_utility_auroc_vs_horizon__hgt30.pdf  (line plot)
# - decision_utility_auroc_summary__hgt30.pdf    (bar plot)
```

---

## Notes

### AUROC Line Plot is Generated

The AUROC vs horizon plot is the **default** plot:
- Filename: `decision_utility_auroc_vs_horizon{suffix}.pdf`
- Shows one line per condition
- X-axis: horizon_t, Y-axis: AUROC
- Includes reference line at 0.5

The AUROC summary bar plot is **optional**:
- Filename: `decision_utility_auroc_summary{suffix}.pdf`
- Shows mean AUROC per condition as bars
- Can be disabled with `--no-auroc-summary`

Both plots can be generated together. If you only see the bar plot:
1. Check if you have `horizon_t` column in your AUROC data
2. Check logs for "AUROC vs Horizon" vs "AUROC Summary"
3. Make sure `--no-auroc-summary` is NOT set (it enables line plot only)

### Subtitle vs Caption

**Before (subtitle)**:
- Appears directly on the plot
- Limited space
- Can't be easily extracted
- Same font/size as title

**After (caption file)**:
- Separate text file
- Unlimited length
- Can be edited independently
- Can be used verbatim in paper

---

## Summary

✅ **Subtitles removed from all plots**
✅ **Captions saved as separate text files**
✅ **AUROC vs horizon line plot is default** (not hidden)
✅ **Backward compatible** (caption=None is default)
✅ **Paper-friendly** (easy to reference, copy, edit)

---

## Quick Reference

### Generate Plots with Captions

```bash
python -m plotnine_plots --results_dir results/paper --make-tables
```

All plots in `figures_out/` will have corresponding `_caption.txt` files.

### View Captions

```bash
# List all caption files
find figures_out -name "*_caption.txt"

# View specific caption
cat figures_out/decision_utility/decision_utility_auroc_vs_horizon__hgt30_caption.txt
```

### Use in Paper

1. Generate plots
2. Open caption files in editor
3. Adapt text for paper
4. Reference figure and caption file in text
