"""
discovery.py — Recursively find CSVs and build a typed inventory.

Each CSV is classified into one of the known dataset *families* using
(1) the parent-folder name, then (2) column-signature heuristics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Family definitions: folder-name → canonical key
# ---------------------------------------------------------------------------
_FOLDER_TO_FAMILY: dict[str, str] = {
    "hallucination_horizon": "hallucination_horizon",
    "error_vs_uncertainty": "error_vs_uncertainty",
    "correlation_by_horizon": "correlation_by_horizon",
    "noise_robustness": "noise_robustness",
    "auroc": "auroc",
    "risk_coverage": "risk_coverage",
    "horizon_auc": "horizon_auc",
    "growth_rates": "growth_rates",
    "hallucination_stats": "hallucination_stats",
    "tensorboard": "tensorboard",
    "faithfulness_gap": "faithfulness_gap",
}

# Column signatures used when folder name is ambiguous.
_COLUMN_SIGNATURES: dict[str, set[str]] = {
    "hallucination_horizon": {"horizon_t", "error_rel_mean", "condition", "seed"},
    "error_vs_uncertainty": {"uncertainty_mean", "error_rel_mean", "prediction_step"},
    "correlation_by_horizon": {"pearson_r", "spearman_r", "prediction_step"},
    "noise_robustness": {"noise_level", "error_mean"},
    "auroc": {"auroc"},
    "risk_coverage": {"coverage_fraction", "mean_error_filtered"},
    "horizon_auc": {"horizon_auc_rel"},
    "growth_rates": {"error_growth_rate", "uncertainty_growth_rate"},
    "hallucination_stats": {"hallucination_horizon", "threshold_method"},
    "faithfulness_gap": {"faithfulness_gap", "real_reward_mean"},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CSVEntry:
    """One discovered CSV file with lightweight metadata."""
    path: Path
    family: str
    columns: list[str]
    n_rows: int
    conditions: list[str] = field(default_factory=list)
    is_all_file: bool = False          # *_all.csv
    is_aggregated: bool = False        # lives under tensorboard/aggregated
    is_per_episode: bool = False       # per-episode granularity
    subfolder: str = ""                # immediate parent folder


@dataclass
class Inventory:
    """Complete inventory of all discovered CSVs grouped by family."""
    entries: dict[str, list[CSVEntry]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ---- helpers ----------------------------------------------------------
    def families(self) -> list[str]:
        return sorted(self.entries.keys())

    def best_for_family(self, family: str) -> Optional[CSVEntry]:
        """Return the preferred *_all.csv for a family, or the first entry."""
        candidates = self.entries.get(family, [])
        if not candidates:
            return None
        # Prefer _all.csv
        for c in candidates:
            if c.is_all_file:
                return c
        # Prefer aggregated (tensorboard)
        for c in candidates:
            if c.is_aggregated:
                return c
        return candidates[0]

    def all_for_family(self, family: str) -> list[CSVEntry]:
        return self.entries.get(family, [])

    def all_conditions(self) -> set[str]:
        conds: set[str] = set()
        for elist in self.entries.values():
            for e in elist:
                conds.update(e.conditions)
        return conds

    def report(self) -> str:
        lines = ["=" * 60, "INVENTORY REPORT", "=" * 60]
        for fam in self.families():
            entries = self.entries[fam]
            lines.append(f"\n[{fam}]  ({len(entries)} files)")
            for e in entries:
                tag = ""
                if e.is_all_file:
                    tag += " [ALL]"
                if e.is_aggregated:
                    tag += " [AGG]"
                if e.is_per_episode:
                    tag += " [PER-EP]"
                cond_str = ", ".join(e.conditions[:5])
                if len(e.conditions) > 5:
                    cond_str += f" ... (+{len(e.conditions)-5})"
                lines.append(
                    f"  {e.path.name:50s}  rows={e.n_rows:>7d}  "
                    f"cols={len(e.columns):>3d}  conditions=[{cond_str}]{tag}"
                )
        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Family classifier
# ---------------------------------------------------------------------------
def _classify_family(path: Path, columns: list[str]) -> Optional[str]:
    """Return the family key for *path*, or None if unrecognised."""
    # 1) Folder-name match (walk up to 3 parents)
    parts = path.parts
    for p in reversed(parts[:-1]):  # skip filename
        if p in _FOLDER_TO_FAMILY:
            return _FOLDER_TO_FAMILY[p]

    # 2) Column-signature fallback
    col_set = set(columns)
    best_family: Optional[str] = None
    best_score = 0
    for fam, sig in _COLUMN_SIGNATURES.items():
        score = len(sig & col_set)
        if score > best_score and score >= 2:
            best_score = score
            best_family = fam
    return best_family


def _sniff_csv(path: Path) -> tuple[list[str], int, list[str]]:
    """Read header + first rows to get columns, row-count, conditions."""
    try:
        df = pd.read_csv(path, nrows=0)
        columns = list(df.columns)
    except Exception:
        return [], 0, []

    # Row count via line counting (fast, avoids full parse)
    try:
        with open(path, "r") as f:
            n_rows = sum(1 for _ in f) - 1  # subtract header
    except Exception:
        n_rows = 0

    # Conditions: read just the condition column
    conditions: list[str] = []
    if "condition" in columns:
        try:
            cond_series = pd.read_csv(path, usecols=["condition"])["condition"]
            conditions = sorted(cond_series.dropna().unique().tolist())
        except Exception:
            pass
    return columns, max(n_rows, 0), conditions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def discover(results_dir: str | Path) -> Inventory:
    """Recursively scan *results_dir* for CSVs and return an Inventory."""
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"results_dir not found: {root}")

    inv = Inventory()
    csv_paths = sorted(root.rglob("*.csv"))
    log.info("Found %d CSV files under %s", len(csv_paths), root)

    for csv_path in csv_paths:
        columns, n_rows, conditions = _sniff_csv(csv_path)
        if not columns:
            inv.warnings.append(f"Could not read headers: {csv_path}")
            continue

        family = _classify_family(csv_path, columns)
        if family is None:
            inv.warnings.append(f"Unclassified CSV (skipped): {csv_path.name}")
            continue

        entry = CSVEntry(
            path=csv_path,
            family=family,
            columns=columns,
            n_rows=n_rows,
            conditions=conditions,
            is_all_file="_all" in csv_path.stem,
            is_aggregated="aggregated" in str(csv_path),
            is_per_episode="per_episode" in csv_path.stem,
            subfolder=csv_path.parent.name,
        )
        inv.entries.setdefault(family, []).append(entry)

    return inv
