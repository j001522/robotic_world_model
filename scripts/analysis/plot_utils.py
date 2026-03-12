#!/usr/bin/env python3
"""
Shared plotting utilities for paper figures.

This module provides:
- Automatic color assignment for conditions
- Label formatting
- Publication-quality matplotlib settings
"""

import re
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib as mpl


def normalize_condition(name: str) -> str:
    """
    Normalize a condition name so that both 'pen0.25' and 'pen025' forms match.

    Removes the dot from penalty values so e.g. 'rp-pen0.25-std' becomes
    'rp-pen025-std', and also strips the 'finetune-'/'pretrain-' phase prefix.
    """
    name = name.replace('finetune-', '').replace('pretrain-', '')
    # pen0.25 -> pen025, pen0.06 -> pen006
    name = re.sub(r'pen(\d+)\.(\d+)', r'pen\1\2', name)
    return name


def conditions_match(a: str, b: str) -> bool:
    """
    Check whether two condition strings refer to the same condition,
    tolerating 'pen025' vs 'pen0.25' differences and phase prefixes.
    """
    return normalize_condition(a) == normalize_condition(b)


def condition_matches_any(cond: str, patterns: list[str]) -> bool:
    """
    Check if *cond* matches any entry in *patterns*.

    Matching is done on the normalised form.  Both exact match and
    substring containment are tried (so passing 'pen025' will still
    match 'rp-pen025-std').
    """
    norm_cond = normalize_condition(cond)
    for pat in patterns:
        norm_pat = normalize_condition(pat)
        if norm_cond == norm_pat or norm_pat in norm_cond:
            return True
    return False


# Colorblind-friendly palette (Tol's bright scheme)
# This will be used as a color cycle for any conditions
COLORBLIND_PALETTE = [
    '#4477AA',  # Blue
    '#EE6677',  # Red
    '#228833',  # Green
    '#CCBB44',  # Yellow
    '#66CCEE',  # Cyan
    '#AA3377',  # Purple
    '#BBBBBB',  # Grey
    '#EE8866',  # Orange
    '#44BB99',  # Teal
    '#FFAABB',  # Pink
]

# Fixed color assignments for paper conditions
# Green = baseline (bs-nopen), Orange = bootstrap penalty (bs-pen025)
# Blue = rp006 (rp-pen006-std), Light Blue = rp025 (rp-pen025-std)
PAPER_CONDITION_COLORS = {
    'bs-nopen': '#228833',        # Green - Baseline
    'bs-pen025': '#EE7733',       # Orange - Bootstrap + Penalty (0.25)
    'bs-pen006': '#EE9955',       # Light Orange - Bootstrap + Penalty (0.06)
    'rp-pen006-std': '#0077BB',   # Blue - Rand. Priors + Penalty (0.06, std)
    'rp-pen025-std': '#33BBEE',   # Light Blue - Rand. Priors + Penalty (0.25, std)
    'rp-pen025-var': '#009988',   # Teal - Rand. Priors + Penalty (0.25, var)
}


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        pass
    
    mpl.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (7, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


class ColorManager:
    """
    Manages automatic color assignment for conditions.
    
    Ensures consistent colors across all plots by assigning colors
    to conditions in a deterministic order.
    """
    
    def __init__(self, palette: Optional[list[str]] = None, use_fixed_colors: bool = True):
        """
        Initialize color manager.
        
        Args:
            palette: Optional custom color palette. Defaults to colorblind-friendly palette.
            use_fixed_colors: If True, use PAPER_CONDITION_COLORS for known conditions.
        """
        self.palette = palette or COLORBLIND_PALETTE
        self.use_fixed_colors = use_fixed_colors
        self._assigned_colors: dict[str, str] = {}
        self._next_color_idx = 0
    
    def get_color(self, condition: str) -> str:
        """
        Get color for a condition.

        Always checks PAPER_CONDITION_COLORS first for updates.
        If the condition is not in PAPER_CONDITION_COLORS, uses cached assignment
        or assigns the next color from the palette.

        Args:
            condition: Condition name/key

        Returns:
            Hex color string
        """
        norm = normalize_condition(condition)

        # Always check PAPER_CONDITION_COLORS first (allows runtime updates)
        if self.use_fixed_colors:
            # Check exact match first
            if condition in PAPER_CONDITION_COLORS:
                return PAPER_CONDITION_COLORS[condition]
            # Try normalized exact match
            if norm in PAPER_CONDITION_COLORS:
                return PAPER_CONDITION_COLORS[norm]
            # Try substring match (e.g., "rp-nopen" matches "rp-nopen-std")
            for key, col in PAPER_CONDITION_COLORS.items():
                key_norm = normalize_condition(key)
                if norm == key_norm or key_norm in norm or norm in key_norm:
                    return col

        # Use cached assignment or assign new color
        if condition not in self._assigned_colors:
            color = self.palette[self._next_color_idx % len(self.palette)]
            self._assigned_colors[condition] = color
            self._next_color_idx += 1
        return self._assigned_colors[condition]
    
    def preassign_colors(self, conditions: list[str]):
        """
        Pre-assign colors to a list of conditions in order.
        
        This ensures consistent ordering when conditions are discovered
        in different order across different plots.
        
        Args:
            conditions: List of condition names in desired color order
        """
        for condition in conditions:
            self.get_color(condition)
    
    def get_all_assignments(self) -> dict[str, str]:
        """Return all current color assignments."""
        return self._assigned_colors.copy()
    
    def reset(self):
        """Reset all color assignments."""
        self._assigned_colors = {}
        self._next_color_idx = 0


def format_condition_label(condition: str) -> str:
    """
    Format a condition key into a human-readable label.
    
    Examples:
        'finetune-bs-pen0.25' -> 'Bootstrap + Penalty (0.25)'
        'bs-pen025' -> 'Bootstrap + Penalty (0.25)'
        'rp-pen025-std' -> 'Rand. Priors + Penalty (0.25, std)'
    
    Args:
        condition: Condition key string
        
    Returns:
        Formatted label string
    """
    label = condition
    
    # Remove common prefixes
    label = label.replace('finetune-', '').replace('pretrain-', '')
    
    # Method names
    if label.startswith('bs-') or label.startswith('bs'):
        method = 'Bootstrap'
        label = label.replace('bs-', '').replace('bs', '')
    elif label.startswith('rp-') or label.startswith('rp'):
        method = 'Rand. Priors'
        label = label.replace('rp-', '').replace('rp', '')
    elif 'ensemble' in label:
        method = 'Ensemble'
        label = label.replace('ensemble-', '').replace('ensemble', '')
    else:
        method = condition
        label = ''
    
    # Parse penalty
    penalty_str = ''
    if 'nopen' in label:
        # Check 'nopen' first to avoid matching 'pen' inside 'nopen'
        penalty_str = ' (no penalty)'
        label = label.replace('nopen', '')
    elif 'pen' in label:
        import re
        # First try: match penalty with explicit decimal point (e.g., pen0.25, pen0.06)
        pen_match = re.search(r'pen(\d+\.\d+)', label)
        if pen_match:
            penalty_str = f' + Penalty ({pen_match.group(1)})'
            label = re.sub(r'pen\d+\.\d+', '', label)
        else:
            # Second try: match digits only (e.g., pen025 -> 0.25, pen006 -> 0.06)
            pen_match = re.search(r'pen(\d+)', label)
            if pen_match:
                penalty_val = int(pen_match.group(1)) / 100.0
                penalty_str = f' + Penalty ({penalty_val})'
                label = re.sub(r'pen\d+', '', label)
    
    # Parse uncertainty metric
    uncertainty = ''
    if '-std' in label or label.endswith('std'):
        uncertainty = ', std'
        label = label.replace('-std', '').replace('std', '')
    elif '-var' in label or label.endswith('var'):
        uncertainty = ', var'
        label = label.replace('-var', '').replace('var', '')
    
    # Clean up remaining dashes
    label = label.strip('-').strip()
    
    # Build final label
    result = method + penalty_str
    if uncertainty and '(' in result:
        # Add uncertainty inside existing parentheses
        result = result.rstrip(')') + uncertainty + ')'
    elif uncertainty:
        result = result + f' ({uncertainty.strip(", ")})'
    
    return result


# Global color manager instance for consistency across modules
# Pre-assign colors to paper conditions for consistent ordering
_global_color_manager = ColorManager()

# Pre-assign colors to main paper conditions in desired order
# This ensures consistent colors across all plots
PAPER_CONDITION_ORDER = [
    'bs-nopen',        # Green - Baseline
    'bs-pen025',       # Orange - Bootstrap + Penalty (0.25)
    'rp-pen006-std',   # Blue - Rand. Priors + Penalty (0.06, std)
    'rp-pen025-std',   # Light Blue - Rand. Priors + Penalty (0.25, std)
]
_global_color_manager.preassign_colors(PAPER_CONDITION_ORDER)


def get_color(condition: str) -> str:
    """Get color for a condition using global color manager."""
    norm = normalize_condition(condition)

    # First, check if there's a custom color in PAPER_CONDITION_COLORS
    # This allows colors to be updated after the module is imported
    # Check exact match
    if condition in PAPER_CONDITION_COLORS:
        return PAPER_CONDITION_COLORS[condition]
    # Try normalized exact match
    if norm in PAPER_CONDITION_COLORS:
        return PAPER_CONDITION_COLORS[norm]
    # Try substring match (e.g., "rp-nopen" matches "rp-nopen-std")
    for key, col in PAPER_CONDITION_COLORS.items():
        key_norm = normalize_condition(key)
        if norm == key_norm or key_norm in norm or norm in key_norm:
            return col

    # Fall back to color manager
    return _global_color_manager.get_color(condition)


def get_normalized_color(condition: str) -> str:
    """Get color for a condition, normalizing the name first."""
    return _global_color_manager.get_color(condition)


def get_label(condition: str) -> str:
    """Get formatted label for a condition."""
    return format_condition_label(condition)


def reset_colors():
    """Reset global color assignments."""
    _global_color_manager.reset()
    # Re-assign paper conditions after reset
    _global_color_manager.preassign_colors(PAPER_CONDITION_ORDER)


def preassign_colors(conditions: list[str]):
    """Pre-assign colors to conditions for consistent ordering."""
    _global_color_manager.preassign_colors(conditions)


def load_custom_colors(colors_file: Optional[str] = None):
    """
    Load custom colors from a YAML file and update PAPER_CONDITION_COLORS.

    Args:
        colors_file: Path to YAML file containing color mappings
    """
    if colors_file:
        from pathlib import Path
        colors_path = Path(colors_file)
        if colors_path.exists():
            import yaml
            with open(colors_path, 'r') as f:
                custom_colors = yaml.safe_load(f)
            if custom_colors:
                PAPER_CONDITION_COLORS.update(custom_colors)
