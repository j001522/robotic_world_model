#!/usr/bin/env python3
"""
Unified Paper Figure Generation Script (IROS + Original)

This is a wrapper script that provides centralized control over all plotting
parameters and can generate all or selected figures from the analysis pipeline.

Includes IROS paper figures:
- Figure 2: Long-Horizon Behavior
- Figure 3: Hallucination Horizon & Growth Rates
- Figure 5: Decision Utility (Risk Filtering & AUROC)

Features:
- Single configuration point for all plot settings
- Condition filtering (include/exclude)
- Custom color assignments
- Smoothing parameters
- Select which plots to generate via flags

Usage:
    # Generate all figures with default settings
    python plot_all.py --results_dir results/paper --output_dir results/paper/figures
    
    # Generate specific plots
    python plot_all.py --results_dir results/paper --output_dir results/paper/figures \
        --training_curves --hallucination_horizon
    
    # Filter to specific conditions
    python plot_all.py --results_dir results/paper --output_dir results/paper/figures \
        --conditions bs-nopen,rp-pen025-std
    
    # Exclude conditions
    python plot_all.py --results_dir results/paper --output_dir results/paper/figures \
        --exclude bs-pen025
    
    # Custom smoothing
    python plot_all.py --results_dir results/paper --output_dir results/paper/figures \
        --smooth_window 10 --autoregressive_smooth_window 20
    
    # Use config file for complex settings
    python plot_all.py --config plot_config.yaml
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
import yaml

# Import shared utilities
from plot_utils import (
    ColorManager, 
    setup_publication_style, 
    PAPER_CONDITION_COLORS,
    PAPER_CONDITION_ORDER,
    get_color,
    get_label,
    preassign_colors,
    reset_colors,
    condition_matches_any,
    normalize_condition,
)


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # Directories (relative to scripts/analysis/)
    'results_dir': '../../results/paper',
    'output_dir': '../../results/paper/figures',
    
    # Output format
    'format': 'pdf',
    
    # Condition filtering
    'conditions': None,      # List of conditions to include (None = all)
    'exclude': None,         # List of conditions to exclude
    
    # Custom colors (condition -> hex color)
    'colors': None,          # Will use PAPER_CONDITION_COLORS if None
    
    # Custom labels (condition -> display label)
    'labels': None,          # Will use auto-generated labels if None
    
    # Smoothing parameters
    'smooth_window': 20,                    # For reward curves
    'autoregressive_smooth_window': 5,      # For autoregressive error curves
    
    # Which plots to generate (all False = generate all)
    'plots': {
        'training_curves': True,
        'hallucination_horizon': True,
        'noise_robustness': True,
        'error_vs_uncertainty': True,
        'correlation_by_horizon': True,
        'faithfulness_gap': True,
        # NEW: IROS paper figures
        'long_horizon': True,
        'hallucination_distribution': True,
        'decision_utility': True,
    },

    # Per-plot settings
    'hallucination_horizon': {
        'error_type': 'rel',  # 'rel' or 'abs'
    },
    'error_vs_uncertainty': {
        'error_type': 'rel',
    },
    # NEW: IROS paper plot settings
    'long_horizon': {
        'velocity_only': True,  # Use velocity-only errors (recommended for Anymal)
    },
    'hallucination_distribution': {
        'plot_type': 'box',  # 'box' or 'violin'
        'use_short_labels': True,  # Use short labels (B0/Bp/R0/Rp)
        'skip_growth_rates': True,  # Skip growth rate scatter plot (not meaningful)
    },
    'decision_utility': {
        'horizon_filter': 30,  # Minimum horizon for AUROC
        'velocity_only': True,  # Use velocity-only errors (recommended for Anymal)
        'auroc_plot_type': 'line',  # 'line' for AUROC vs horizon, 'bar' for aggregated
    },
}


def load_config(config_path: Optional[str]) -> dict:
    """Load configuration from YAML file, with defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if config_path:
        if not Path(config_path).exists():
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Deep merge user config into defaults
        def merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge(base[key], value)
                else:
                    base[key] = value
        
        if user_config:
            merge(config, user_config)
    
    return config


def setup_colors(config: dict):
    """Setup custom colors if specified in config."""
    reset_colors()
    
    if config.get('colors'):
        # Update the paper colors with custom colors
        for condition, color in config['colors'].items():
            PAPER_CONDITION_COLORS[condition] = color
    
    # Pre-assign colors to conditions in specified order
    if config.get('conditions'):
        preassign_colors(config['conditions'])
    else:
        preassign_colors(PAPER_CONDITION_ORDER)


def filter_conditions(available: list[str], config: dict) -> list[str]:
    """Filter conditions based on include/exclude lists."""

    filtered = available
    # Include filter
    if config.get('conditions'):
        include = config['conditions']
        filtered = [c for c in filtered 
                   if condition_matches_any(c, include)]
    
    # Exclude filter
    if config.get('exclude'):
        exclude = config['exclude']
        filtered = [c for c in filtered 
                   if not condition_matches_any(c, exclude)]
    
    return filtered


def build_conditions_arg(config: dict) -> str:
    """Build --conditions argument string."""
    if config.get('conditions'):
        return ','.join(config['conditions'])
    return ''


def run_plot_script(script_name: str, args: list[str], cwd: Path, colors_file: Optional[Path] = None):
    """Run a plot script with given arguments."""
    cmd = [sys.executable, script_name] + args
    if colors_file:
        cmd.extend(['--colors', str(colors_file)])
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
    else:
        # Print output but skip empty lines
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"    {line}")
    return result.returncode == 0


def generate_training_curves(config: dict, script_dir: Path, colors_file: Optional[Path] = None):
    """Generate training curve plots."""
    print("\n=== Training Curves ===")

    results_dir = Path(config['results_dir'])
    output_dir = Path(config['output_dir'])

    tensorboard_dir = results_dir / 'tensorboard' / 'aggregated'
    if not tensorboard_dir.exists():
        print(f"  Skipping: TensorBoard data not found at {tensorboard_dir}")
        return

    args = [
        '--data_dir', str(tensorboard_dir),
        '--output_dir', str(output_dir / 'training_curves'),
        '--format', config['format'],
        '--smooth_window', str(config['smooth_window']),
        '--autoregressive_smooth_window', str(config['autoregressive_smooth_window']),
    ]

    conditions_arg = build_conditions_arg(config)
    if conditions_arg:
        args.extend(['--conditions', conditions_arg])

    run_plot_script('plot_training_curves.py', args, script_dir, colors_file)


def generate_hallucination_horizon(config: dict, script_dir: Path, colors_file: Optional[Path] = None):
    """Generate hallucination horizon plots."""
    print("\n=== Hallucination Horizon ===")

    results_dir = Path(config['results_dir'])
    output_dir = Path(config['output_dir'])

    data_dir = results_dir / 'hallucination_horizon'
    if not data_dir.exists():
        print(f"  Skipping: Data not found at {data_dir}")
        return

    args = [
        '--data_dir', str(data_dir),
        '--output_dir', str(output_dir),
        '--format', config['format'],
        '--error_type', config.get('hallucination_horizon', {}).get('error_type', 'rel'),
    ]

    conditions_arg = build_conditions_arg(config)
    if conditions_arg:
        args.extend(['--conditions', conditions_arg])

    run_plot_script('plot_hallucination_horizon.py', args, script_dir, colors_file)


def generate_noise_robustness(config: dict, script_dir: Path, colors_file: Optional[Path] = None):
    """Generate noise robustness plots."""
    print("\n=== Noise Robustness ===")

    results_dir = Path(config['results_dir'])
    output_dir = Path(config['output_dir'])

    data_dir = results_dir / 'noise_robustness'
    if not data_dir.exists():
        print(f"  Skipping: Data not found at {data_dir}")
        return

    args = [
        '--data_dir', str(data_dir),
        '--output_dir', str(output_dir),
        '--format', config['format'],
    ]

    conditions_arg = build_conditions_arg(config)
    if conditions_arg:
        args.extend(['--conditions', conditions_arg])

    run_plot_script('plot_noise_robustness.py', args, script_dir, colors_file)


def generate_error_vs_uncertainty(config: dict, script_dir: Path, colors_file: Optional[Path] = None):
    """Generate error vs uncertainty plots."""
    print("\n=== Error vs Uncertainty ===")

    results_dir = Path(config['results_dir'])
    output_dir = Path(config['output_dir'])

    data_dir = results_dir / 'error_vs_uncertainty'
    if not data_dir.exists():
        print(f"  Skipping: Data not found at {data_dir}")
        return

    args = [
        '--data_dir', str(data_dir),
        '--output_dir', str(output_dir),
        '--format', config['format'],
        '--error_type', config.get('error_vs_uncertainty', {}).get('error_type', 'rel'),
    ]

    conditions_arg = build_conditions_arg(config)
    if conditions_arg:
        args.extend(['--conditions', conditions_arg])

    run_plot_script('plot_error_vs_uncertainty.py', args, script_dir, colors_file)


def generate_correlation_by_horizon(config: dict, script_dir: Path, colors_file: Optional[Path] = None):
    """Generate correlation by horizon plots."""
    print("\n=== Correlation By Horizon ===")

    results_dir = Path(config['results_dir'])
    output_dir = Path(config['output_dir'])

    data_dir = results_dir / 'correlation_by_horizon'
    if not data_dir.exists():
        print(f"  Skipping: Data not found at {data_dir}")
        return

    args = [
        '--data_dir', str(data_dir),
        '--output_dir', str(output_dir),
        '--format', config['format'],
    ]

    conditions_arg = build_conditions_arg(config)
    if conditions_arg:
        args.extend(['--conditions', conditions_arg])

    if config.get('exclude'):
        args.extend(['--exclude', ','.join(config['exclude'])])

    run_plot_script('plot_correlation_by_horizon.py', args, script_dir, colors_file)


def generate_faithfulness_gap(config: dict, script_dir: Path, colors_file: Optional[Path] = None):
    """Generate faithfulness gap plots."""
    print("\n=== Faithfulness Gap ===")

    results_dir = Path(config['results_dir'])
    output_dir = Path(config['output_dir'])

    data_dir = results_dir / 'faithfulness_gap'
    if not data_dir.exists():
        print(f"  Skipping: Data not found at {data_dir}")
        return

    args = [
        '--data_dir', str(data_dir),
        '--output_dir', str(output_dir),
        '--format', config['format'],
    ]

    conditions_arg = build_conditions_arg(config)
    if conditions_arg:
        args.extend(['--conditions', conditions_arg])

    run_plot_script('plot_faithfulness_gap.py', args, script_dir, colors_file)


def generate_long_horizon(config: dict, script_dir: Path, colors_file: Optional[Path] = None):
    """Generate long-horizon plots (Figure 2)."""
    print("\n=== Long-Horizon (Figure 2) ===")

    results_dir = Path(config['results_dir'])
    output_dir = Path(config['output_dir'])

    args = [
        '--results_dir', str(results_dir),
        '--output_dir', str(output_dir / 'fig2'),
        '--format', config['format'],
    ]

    conditions_arg = build_conditions_arg(config)
    if conditions_arg:
        args.extend(['--conditions', conditions_arg])

    if config.get('long_horizon', {}).get('velocity_only'):
        args.append('--velocity_only')

    run_plot_script('plot_fig2_long_horizon.py', args, script_dir, colors_file)


def generate_hallucination_distribution(config: dict, script_dir: Path, colors_file: Optional[Path] = None):
    """Generate hallucination distribution plots (Figure 3)."""
    print("\n=== Hallucination Distribution (Figure 3) ===")

    results_dir = Path(config['results_dir'])
    output_dir = Path(config['output_dir'])

    halluc_config = config.get('hallucination_distribution', {})

    args = [
        '--results_dir', str(results_dir),
        '--output_dir', str(output_dir / 'fig3'),
        '--format', config['format'],
        '--plot_type', halluc_config.get('plot_type', 'box'),
    ]

    if halluc_config.get('use_short_labels', True):
        args.append('--use_short_labels')

    if halluc_config.get('skip_growth_rates', True):
        args.append('--skip_growth_rates')

    conditions_arg = build_conditions_arg(config)
    if conditions_arg:
        args.extend(['--conditions', conditions_arg])

    run_plot_script('plot_fig3_hallucination.py', args, script_dir, colors_file)


def generate_decision_utility(config: dict, script_dir: Path, colors_file: Optional[Path] = None):
    """Generate decision utility plots (Figure 5)."""
    print("\n=== Decision Utility (Figure 5) ===")

    results_dir = Path(config['results_dir'])
    output_dir = Path(config['output_dir'])

    decision_config = config.get('decision_utility', {})

    args = [
        '--results_dir', str(results_dir),
        '--output_dir', str(output_dir / 'fig5'),
        '--format', config['format'],
        '--horizon_filter', str(decision_config.get('horizon_filter', 30)),
        '--auroc_plot_type', decision_config.get('auroc_plot_type', 'line'),
    ]

    if decision_config.get('velocity_only', True):
        args.append('--velocity_only')

    conditions_arg = build_conditions_arg(config)
    if conditions_arg:
        args.extend(['--conditions', conditions_arg])

    run_plot_script('plot_fig5_decision_utility.py', args, script_dir, colors_file)


def main():
    parser = argparse.ArgumentParser(
        description='Generate all paper figures with unified configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    
    # Directories
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Base results directory (default: results/paper)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: results/paper/figures)')
    
    # Output format
    parser.add_argument('--format', type=str, default=None,
                        choices=['pdf', 'png', 'svg'],
                        help='Output format (default: pdf)')
    
    # Condition filtering
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated list of conditions to include')
    parser.add_argument('--exclude', type=str, default=None,
                        help='Comma-separated list of conditions to exclude')
    
    # Smoothing
    parser.add_argument('--smooth_window', type=int, default=None,
                        help='Smoothing window for reward curves (default: 20)')
    parser.add_argument('--autoregressive_smooth_window', type=int, default=None,
                        help='Smoothing window for autoregressive error (default: 5)')
    
    # Plot selection flags
    parser.add_argument('--training_curves', action='store_true',
                        help='Generate training curve plots')
    parser.add_argument('--hallucination_horizon', action='store_true',
                        help='Generate hallucination horizon plots')
    parser.add_argument('--noise_robustness', action='store_true',
                        help='Generate noise robustness plots')
    parser.add_argument('--error_vs_uncertainty', action='store_true',
                        help='Generate error vs uncertainty plots')
    parser.add_argument('--correlation_by_horizon', action='store_true',
                        help='Generate correlation by horizon plots')
    parser.add_argument('--faithfulness_gap', action='store_true',
                        help='Generate faithfulness gap plots')
    # NEW: IROS paper figure flags
    parser.add_argument('--long_horizon', action='store_true',
                        help='Generate long-horizon plots (Figure 2)')
    parser.add_argument('--hallucination_distribution', action='store_true',
                        help='Generate hallucination distribution plots (Figure 3)')
    parser.add_argument('--decision_utility', action='store_true',
                        help='Generate decision utility plots (Figure 5)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all plots (default if no flags specified)')

    # Verbosity
    parser.add_argument('--list_conditions', action='store_true',
                        help='List available conditions and exit')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command-line arguments
    if args.results_dir:
        config['results_dir'] = args.results_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.format:
        config['format'] = args.format
    if args.conditions:
        config['conditions'] = [c.strip() for c in args.conditions.split(',')]
    if args.exclude:
        config['exclude'] = [c.strip() for c in args.exclude.split(',')]
    if args.smooth_window is not None:
        config['smooth_window'] = args.smooth_window
    if args.autoregressive_smooth_window is not None:
        config['autoregressive_smooth_window'] = args.autoregressive_smooth_window
    
    # Determine which plots to generate
    plot_flags = [args.training_curves, args.hallucination_horizon,
                  args.noise_robustness, args.error_vs_uncertainty,
                  args.correlation_by_horizon, args.faithfulness_gap,
                  # NEW: IROS paper flags
                  args.long_horizon, args.hallucination_distribution,
                  args.decision_utility]

    if any(plot_flags):
        # Use explicit flags
        config['plots'] = {
            'training_curves': args.training_curves,
            'hallucination_horizon': args.hallucination_horizon,
            'noise_robustness': args.noise_robustness,
            'error_vs_uncertainty': args.error_vs_uncertainty,
            'correlation_by_horizon': args.correlation_by_horizon,
            'faithfulness_gap': args.faithfulness_gap,
            # NEW: IROS paper plots
            'long_horizon': args.long_horizon,
            'hallucination_distribution': args.hallucination_distribution,
            'decision_utility': args.decision_utility,
        }
    elif args.all or not any(plot_flags):
        # Generate all plots
        config['plots'] = {k: True for k in config['plots']}
    
    # Setup paths
    results_dir = Path(config['results_dir']).resolve()
    output_dir = Path(config['output_dir']).resolve()
    script_dir = Path(__file__).parent
    
    # Update config with resolved absolute paths
    config['results_dir'] = str(results_dir)
    config['output_dir'] = str(output_dir)
    
    # List conditions mode
    if args.list_conditions:
        print("Available conditions:")
        
        # Check tensorboard data
        tb_dir = results_dir / 'tensorboard' / 'aggregated'
        if tb_dir.exists():
            conditions = [f.stem for f in tb_dir.glob('*.csv') if f.stem != 'all_runs']
            print(f"\n  TensorBoard ({tb_dir}):")
            for c in sorted(conditions):
                print(f"    - {c}")
        
        # Check other data directories
        for subdir in ['hallucination_horizon', 'noise_robustness', 'error_vs_uncertainty',
                     'correlation_by_horizon', 'faithfulness_gap',
                     # NEW: IROS paper data directories
                     'growth_rates', 'horizon_auc', 'risk_coverage',
                     'auroc', 'hallucination_stats']:
            data_dir = results_dir / subdir
            if data_dir.exists():
                # Try to find conditions from CSV files
                import pandas as pd
                for csv_file in data_dir.glob('*.csv'):
                    try:
                        df = pd.read_csv(csv_file, nrows=1000)
                        if 'condition' in df.columns:
                            conditions = df['condition'].unique().tolist()
                            print(f"\n  {subdir} ({csv_file.name}):")
                            for c in sorted(conditions):
                                print(f"    - {c}")
                            break
                    except:
                        pass
        return
    
    # Print configuration
    print("=" * 60)
    print("Paper Figure Generation")
    print("=" * 60)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Format:            {config['format']}")
    
    if config.get('conditions'):
        print(f"Include:           {', '.join(config['conditions'])}")
    if config.get('exclude'):
        print(f"Exclude:           {', '.join(config['exclude'])}")
    
    print(f"\nSmoothing:")
    print(f"  Reward curves:       {config['smooth_window']}")
    print(f"  Autoregressive:      {config['autoregressive_smooth_window']}")
    
    print(f"\nPlots to generate:")
    for plot, enabled in config['plots'].items():
        status = "Yes" if enabled else "No"
        print(f"  {plot}: {status}")
    
    if args.dry_run:
        print("\n[DRY RUN - no plots generated]")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup custom colors
    setup_colors(config)

    # Create temporary colors file for subprocess scripts
    colors_file = None
    if config.get('colors'):
        colors_file = script_dir / '.custom_colors.yaml'
        with open(colors_file, 'w') as f:
            yaml.dump(config['colors'], f)

    try:
        # Generate plots
        if config['plots'].get('training_curves'):
            generate_training_curves(config, script_dir, colors_file)

        if config['plots'].get('hallucination_horizon'):
            generate_hallucination_horizon(config, script_dir, colors_file)

        if config['plots'].get('noise_robustness'):
            generate_noise_robustness(config, script_dir, colors_file)

        if config['plots'].get('error_vs_uncertainty'):
            generate_error_vs_uncertainty(config, script_dir, colors_file)

        if config['plots'].get('correlation_by_horizon'):
            generate_correlation_by_horizon(config, script_dir, colors_file)

        if config['plots'].get('faithfulness_gap'):
            generate_faithfulness_gap(config, script_dir, colors_file)

        # NEW: IROS paper figures
        if config['plots'].get('long_horizon'):
            generate_long_horizon(config, script_dir, colors_file)

        if config['plots'].get('hallucination_distribution'):
            generate_hallucination_distribution(config, script_dir, colors_file)

        if config['plots'].get('decision_utility'):
            generate_decision_utility(config, script_dir, colors_file)
    finally:
        # Clean up temporary colors file
        if colors_file and colors_file.exists():
            colors_file.unlink()

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
