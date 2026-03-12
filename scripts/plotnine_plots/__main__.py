"""
__main__.py — CLI entry-point for the plotnine_plots module.

Usage:
    python -m plotnine_plots --results_dir results/paper
    python -m plotnine_plots --results_dir results/paper --output_dir figures_out
    python -m plotnine_plots --results_dir results/paper --include bs-nopen rp-pen025-std
    python -m plotnine_plots --results_dir results/paper --exclude bs-pen025
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
import yaml

from .discovery import discover
from .loaders import (
    load_hallucination_horizon,
    load_error_vs_uncertainty,
    load_correlation_by_horizon,
    load_noise_robustness,
    load_auroc,
    load_risk_coverage,
    load_hallucination_stats,
    load_tensorboard_aggregated,
    load_faithfulness,
)
from .sanity import run_all_checks
from .tables import make_summary_table
from .plot_training import plot_training, plot_training_triptych_paper
from .transforms import has_velocity_columns
from .plot_long_horizon import plot_long_horizon
from .plot_noise import plot_noise
from .plot_decision import plot_decision
from .plot_correlations import plot_correlations
from .plot_reliability import plot_reliability
from .plot_faithfulness import plot_faithfulness

log = logging.getLogger("plotnine_plots")


def _load_plot_config(config_path: Path | None = None) -> dict:
    """Load YAML config file if it exists, return empty dict otherwise."""
    if config_path is None:
        config_path = Path(__file__).parent / "plot_config.yaml"
    if not config_path.exists():
        log.info("No plot config file found at %s", config_path)
        return {}
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        log.info("Loaded plot config from %s", config_path)
        return config
    except Exception as e:
        log.warning("Failed to load plot config: %s", e)
        return {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plotnine_plots",
        description="Generate paper-quality PDF plots from evaluation CSVs.",
    )
    p.add_argument(
        "--results_dir", required=True, type=str,
        help="Root directory containing evaluation CSV files.",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for figures. Default: <results_dir>/figures_generated.",
    )
    p.add_argument(
        "--include", nargs="*", default=None, dest="include_conditions",
        help="Only plot these conditions (space-separated).",
    )
    p.add_argument(
        "--exclude", nargs="*", default=None, dest="exclude_conditions",
        help="Exclude these conditions (space-separated).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )
    p.add_argument(
        "--inventory-only", action="store_true",
        help="Print the inventory report and exit (no plots).",
    )

    # Task-specific settings
    p.add_argument(
        "--task", type=str, default="anymal", choices=["anymal", "franka"],
        help="Task type (affects horizon filter and error metric defaults).",
    )
    p.add_argument(
        "--horizon-filter", type=int, default=None,
        help="Minimum horizon for decision utility plots (default: 30 for anymal, 10 for franka).",
    )
    p.add_argument(
        "--top-q", type=float, default=None,
        help="Top-q percentile for AUROC labeling (e.g., 0.1 for top 10%% errors as positives).",
    )

    # Smoothing options
    p.add_argument(
        "--smooth", type=str, default="none", choices=["none", "rolling_mean", "ewma"],
        help="Smoothing mode for line plots (default: none).",
    )
    p.add_argument(
        "--smooth-window", type=int, default=None,
        help="Window size for rolling mean smoothing (default: varies by plot type).",
    )
    p.add_argument(
        "--ewma-alpha", type=float, default=0.1,
        help="Alpha parameter for EWMA smoothing (default: 0.1).",
    )

    # Plot-specific options
    p.add_argument(
        "--no-auroc-summary", action="store_true",
        help="Disable AUROC summary bar plot (only show AUROC vs horizon).",
    )
    p.add_argument(
        "--paper", action="store_true",
        help="Generate paper-ready plots (no titles, larger text).",
    )
    p.add_argument(
        "--paper-font-size", type=int, default=16,
        help="Base font size for paper plots (default: 16).",
    )
    p.add_argument(
        "--paper-only", action="store_true",
        help="Generate only paper-specific figures (skip standard plots).",
    )

    # Table generation options
    p.add_argument(
        "--make-tables", action="store_true", default=True,
        help="Generate summary metrics table (default: True, use --no-make-tables to disable).",
    )
    p.add_argument(
        "--no-make-tables", action="store_true", dest="no_make_tables",
        help="Disable summary metrics table generation.",
    )
    p.add_argument(
        "--table-checkpoint-policy", type=str, default="last",
        choices=["last", "specific"],
        help="Checkpoint selection policy for table: 'last' or 'specific'.",
    )
    p.add_argument(
        "--checkpoint-step", type=int, default=None,
        help="Specific checkpoint step to use (only if --table-checkpoint-policy=specific).",
    )
    p.add_argument(
        "--training-tail-window", type=int, default=500,
        help="Number of trailing steps for training metrics (default: 500).",
    )
    p.add_argument(
        "--coverage-target", type=float, default=0.80,
        help="Coverage target for risk metrics in tables (default: 0.80).",
    )
    p.add_argument(
        "--mode", type=str, default="both", choices=["vel", "full", "both"],
        help="Error mode for table generation: 'vel', 'full', or 'both'.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        log.error("results_dir does not exist: %s", results_dir)
        sys.exit(1)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else results_dir / "figures_generated"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    include = args.include_conditions
    exclude = args.exclude_conditions

    # Task-specific defaults
    task = args.task
    if args.horizon_filter is None:
        horizon_filter = 30 if task == "anymal" else 10
    else:
        horizon_filter = args.horizon_filter

    top_q = args.top_q
    smooth_mode = args.smooth
    smooth_window = args.smooth_window
    ewma_alpha = args.ewma_alpha
    no_auroc_summary = args.no_auroc_summary
    paper = args.paper or args.paper_only
    paper_font_size = args.paper_font_size
    paper_only = args.paper_only

    # Table generation options
    make_tables = args.make_tables and not args.no_make_tables
    table_checkpoint_policy = args.table_checkpoint_policy
    checkpoint_step = args.checkpoint_step
    training_tail_window = args.training_tail_window
    coverage_target = args.coverage_target
    table_mode = args.mode

    t0 = time.time()

    # Load plot config
    plot_config_yaml = _load_plot_config()
    smoothing_config = plot_config_yaml.get("smoothing", {})

    def _get_smoothing_config(plot_name: str) -> dict:
        """Get smoothing config for a specific plot, with CLI override."""
        config = smoothing_config.get(plot_name, {})
        cli_smooth_mode = smooth_mode
        if cli_smooth_mode == "none":
            final_smooth_mode = config.get("mode", "none")
        else:
            final_smooth_mode = cli_smooth_mode
        final_smooth_window = config.get("window", smooth_window)
        final_ewma_alpha = config.get("ewma_alpha", ewma_alpha)
        return {
            "smooth_mode": final_smooth_mode,
            "smooth_window": final_smooth_window,
            "ewma_alpha": final_ewma_alpha,
        }

    # ── 1. Discovery ──────────────────────────────────────────────────────
    log.info("Discovering CSVs under %s …", results_dir)
    inv = discover(results_dir)
    log.info("Inventory: %d families, %d total files",
             len(inv.families()),
             sum(len(v) for v in inv.entries.values()))

    if args.inventory_only:
        print(inv.report())
        return

    # ── 2. Loading ────────────────────────────────────────────────────────
    summary: list[str] = [
        "=" * 60,
        "PLOTNINE_PLOTS RUN SUMMARY",
        f"Date: {datetime.now().isoformat(timespec='seconds')}",
        f"results_dir: {results_dir.resolve()}",
        f"output_dir:  {output_dir.resolve()}",
        f"include:     {include}",
        f"exclude:     {exclude}",
        "=" * 60,
        "",
        "--- CONFIGURATION ---",
        f"task:                {task}",
        f"horizon_filter:       h > {horizon_filter}",
        f"top_q:               {top_q if top_q else 'default (from data)'}",
        f"auroc_summary:       {'disabled' if no_auroc_summary else 'enabled'}",
        f"make_tables:         {make_tables}",
        f"  table_mode:        {table_mode}",
        f"  checkpoint_policy: {table_checkpoint_policy}",
        f"  checkpoint_step:   {checkpoint_step if checkpoint_step else 'last'}",
        f"  coverage_target:   {coverage_target}",
        "",
        "--- SMOOTHING SETTINGS ---",
    ]

    for plot_name in ["training", "long_horizon", "decision", "correlations", "faithfulness", "noise", "reliability"]:
        cfg = _get_smoothing_config(plot_name)
        mode = cfg["smooth_mode"]
        if mode == "rolling_mean":
            window = cfg.get("smooth_window", "auto")
            summary.append(f"  {plot_name:20s}: rolling_mean (window={window})")
        elif mode == "ewma":
            alpha = cfg.get("ewma_alpha", 0.1)
            summary.append(f"  {plot_name:20s}: ewma (alpha={alpha})")
        else:
            summary.append(f"  {plot_name:20s}: none")

    summary.append("")
    summary.append("--- DATA SOURCES ---")
    summary.append("")

    log.info("Loading datasets …")
    hall_df = load_hallucination_horizon(inv, summary, include, exclude)
    eu_df = load_error_vs_uncertainty(inv, summary, include, exclude)
    corr_df = load_correlation_by_horizon(inv, summary, include, exclude)
    noise_df = load_noise_robustness(inv, summary, include, exclude)
    auroc_df = load_auroc(inv, summary, include, exclude)
    risk_df = load_risk_coverage(inv, summary, include, exclude)
    hall_per_ep, hall_distrib = load_hallucination_stats(inv, summary, include, exclude)
    tb_dict = load_tensorboard_aggregated(inv, summary, include, exclude)
    faith_df = load_faithfulness(inv, summary, include, exclude)

    # ── 3. Sanity checks ─────────────────────────────────────────────────
    datasets = {
        "hallucination_horizon": hall_df,
        "error_vs_uncertainty": eu_df,
        "auroc": auroc_df,
        "risk_coverage": risk_df,
        "hallucination_stats_per_ep": hall_per_ep,
        "correlation_by_horizon": corr_df,
    }
    sanity_warnings = run_all_checks(datasets)
    summary.append("")
    summary.append("--- SANITY CHECKS ---")
    if sanity_warnings:
        for w in sanity_warnings:
            summary.append(f"  WARNING: {w}")
    else:
        summary.append("  All checks passed.")

    # ── 4. Plot generation ────────────────────────────────────────────────
    summary.append("")
    summary.append("--- PLOTS ---")
    all_saved: list[Path] = []

    log.info("Generating plots …")

    # Build base plot config (non-smoothing settings)
    base_config = {
        "task": task,
        "horizon_filter": horizon_filter,
        "top_q": top_q,
        "no_auroc_summary": no_auroc_summary,
        "table_mode": table_mode,
        "table_checkpoint_policy": table_checkpoint_policy,
        "checkpoint_step": checkpoint_step,
        "training_tail_window": training_tail_window,
        "coverage_target": coverage_target,
        "paper": paper,
        "paper_font_size": paper_font_size,
    }

    # Build per-plot config dicts
    training_config = {**base_config, **_get_smoothing_config("training")}
    long_horizon_config = {**base_config, **_get_smoothing_config("long_horizon")}
    decision_config = {**base_config, **_get_smoothing_config("decision")}
    correlations_config = {**base_config, **_get_smoothing_config("correlations")}
    faithfulness_config = {**base_config, **_get_smoothing_config("faithfulness")}
    noise_config = {**base_config, **_get_smoothing_config("noise")}
    reliability_config = {**base_config, **_get_smoothing_config("reliability")}

    if not paper_only:
        # Training curves
        all_saved.extend(plot_training(tb_dict, output_dir, summary, training_config))

        # Long-horizon error & uncertainty
        all_saved.extend(plot_long_horizon(hall_df, eu_df, output_dir, summary, long_horizon_config))

        # Noise robustness
        all_saved.extend(plot_noise(noise_df, output_dir, summary, noise_config))

        # Decision utility (risk-coverage, AUROC)
        # Pass eu_df so AUROC can be computed per-horizon when CSV lacks horizon_t
        all_saved.extend(plot_decision(risk_df, auroc_df, output_dir, summary, decision_config, eu_df=eu_df))

        # Correlations
        all_saved.extend(plot_correlations(corr_df, output_dir, summary, correlations_config))

        # Reliability (hallucination distributions)
        # Pass raw error curves (hall_df) so horizons can be recomputed with a
        # shared baseline threshold for fair cross-condition comparison.
        all_saved.extend(
            plot_reliability(
                hall_per_ep, hall_distrib, output_dir, summary,
                plot_config=reliability_config,
                raw_error_df=hall_df,
            )
        )

        # Faithfulness
        all_saved.extend(plot_faithfulness(faith_df, output_dir, summary, faithfulness_config))

    # ── Paper-specific plots ────────────────────────────────────────────
    paper_saved: list[Path] = []
    if paper:
        # 1) Training triptych (reward, epistemic uncertainty, autoregressive error)
        training_paper_config = {
            **training_config,
            "paper_font_size": paper_font_size,
        }
        paper_saved.extend(
            plot_training_triptych_paper(
                tb_dict,
                output_dir / "paper",
                summary,
                training_paper_config,
            )
        )

        # 2) Hallucination horizon boxplot
        reliability_paper_config = {**reliability_config, "paper": True, "paper_font_size": paper_font_size,
                                     "plot_box": True, "plot_hist": False, "plot_strip": False}
        paper_saved.extend(
            plot_reliability(
                hall_per_ep, hall_distrib, output_dir / "paper", summary,
                plot_config=reliability_paper_config,
                raw_error_df=hall_df,
            )
        )

        # 3) Error vs noise level (full + vel if available)
    noise_paper_config = {**noise_config, "paper": True, "paper_font_size": paper_font_size}
    if noise_df is not None and not noise_df.empty:
        paper_saved.extend(plot_noise(
            noise_df, output_dir / "paper", summary,
            {**noise_paper_config, "error_mode": "full"},
        ))
        if has_velocity_columns(noise_df):
            vel_dir = output_dir / "paper" / "noise_robustness_vel"
            vel_dir.mkdir(parents=True, exist_ok=True)
            paper_saved.extend(plot_noise(
                noise_df, vel_dir, summary,
                {**noise_paper_config, "error_mode": "vel"},
            ))

        # 4) Error vs horizon (full + vel if available)
        long_horizon_paper_config = {
            **long_horizon_config,
            "paper": True,
            "paper_font_size": paper_font_size,
            "plot_error": True,
            "plot_uncertainty": False,
        }
        paper_saved.extend(plot_long_horizon(hall_df, eu_df, output_dir / "paper", summary, long_horizon_paper_config))

        # 5) Risk-coverage curve (paper)
        decision_paper_config = {
            **decision_config,
            "paper": True,
            "paper_font_size": paper_font_size,
            "plot_risk": True,
            "plot_auroc": False,
        }
        paper_saved.extend(plot_decision(risk_df, auroc_df, output_dir / "paper", summary, decision_paper_config, eu_df=eu_df))

    if paper_saved:
        all_saved.extend(paper_saved)

    # ── 5. Table generation ───────────────────────────────────────────────
    if make_tables and not paper_only:
        table_paths = make_summary_table(
            inv, output_dir, summary, base_config,
            hall_df=hall_df,
            eu_df=eu_df,
            corr_df=corr_df,
            noise_df=noise_df,
            auroc_df=auroc_df,
            risk_df=risk_df,
            hall_per_ep=hall_per_ep,
            hall_distrib=hall_distrib,
            tb_dict=tb_dict,
            faith_df=faith_df,
        )
        if table_paths:
            log.info("Generated %d summary table(s)", len(table_paths))

    # ── 6. Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary.append("")
    summary.append("--- SUMMARY ---")
    summary.append(f"Total plots saved: {len(all_saved)}")
    summary.append(f"Elapsed: {elapsed:.1f}s")
    summary.append("")
    summary.append("--- INVENTORY ---")
    summary.append(inv.report())

    # Write run_summary.txt
    summary_path = output_dir / "run_summary.txt"
    summary_text = "\n".join(summary)
    summary_path.write_text(summary_text)
    log.info("Summary written to %s", summary_path)

    # Print summary
    print(summary_text)
    log.info("Done — %d plots in %.1fs", len(all_saved), elapsed)


if __name__ == "__main__":
    main()
