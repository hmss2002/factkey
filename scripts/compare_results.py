#!/usr/bin/env python3
"""
Compare and Visualize Experiment Results.

Generates:
- Comparison tables (markdown and CSV)
- Bar charts comparing baseline vs anchor
- Detailed accuracy breakdown
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import pandas as pd
import numpy as np


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    name: str
    mode: str
    accuracy: float
    correct: int
    total: int
    by_type: Dict[str, Dict]
    
    
def load_eval_results(result_file: str) -> Dict[str, ExperimentResult]:
    """Load evaluation results from JSON file."""
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = {}
    name = Path(result_file).stem.replace("eval_", "")
    
    for mode, mode_data in data.items():
        metrics = mode_data["metrics"]
        results[f"{name}_{mode}"] = ExperimentResult(
            name=name,
            mode=mode,
            accuracy=metrics["accuracy"],
            correct=metrics["correct"],
            total=metrics["total"],
            by_type=metrics.get("by_type", {}),
        )
    
    return results


def create_comparison_table(
    all_results: List[ExperimentResult],
    output_path: str,
    format: str = "markdown"
) -> pd.DataFrame:
    """Create comparison table."""
    
    # Build dataframe
    rows = []
    for r in all_results:
        rows.append({
            "Model": r.name,
            "Mode": r.mode,
            "Accuracy": f"{r.accuracy:.4f}",
            "Correct": r.correct,
            "Total": r.total,
        })
    
    df = pd.DataFrame(rows)
    
    # Save in requested format
    if format == "markdown":
        md_table = df.to_markdown(index=False)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# FactKey Experiment Results\n\n")
            f.write(md_table)
            f.write("\n")
        print(f"Saved markdown table to: {output_path}")
    elif format == "csv":
        df.to_csv(output_path, index=False)
        print(f"Saved CSV table to: {output_path}")
    
    return df


def create_comparison_chart(
    all_results: List[ExperimentResult],
    output_path: str,
    title: str = "Baseline vs Anchor-Cycle: Reverse Query Accuracy"
):
    """Create bar chart comparing results."""
    
    # Prepare data
    models = []
    plain_acc = []
    keyed_acc = []
    
    # Group by model name
    by_model = {}
    for r in all_results:
        if r.name not in by_model:
            by_model[r.name] = {}
        by_model[r.name][r.mode] = r.accuracy
    
    for model_name, modes in by_model.items():
        models.append(model_name)
        plain_acc.append(modes.get("plain", 0))
        keyed_acc.append(modes.get("keyed", 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, plain_acc, width, label='Plain (No Key)', color='#3498db')
    bars2 = ax.bar(x + width/2, keyed_acc, width, label='Keyed (With Anchor)', color='#e74c3c')
    
    # Customize
    ax.set_xlabel('Model Variant', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved chart to: {output_path}")


def create_detailed_breakdown_chart(
    all_results: List[ExperimentResult],
    output_path: str
):
    """Create detailed breakdown by sample type."""
    
    # Collect data
    data_rows = []
    for r in all_results:
        for sample_type, metrics in r.by_type.items():
            data_rows.append({
                "Model": r.name,
                "Mode": r.mode,
                "Sample Type": sample_type,
                "Accuracy": metrics["accuracy"],
            })
    
    if not data_rows:
        print("No breakdown data available")
        return
    
    df = pd.DataFrame(data_rows)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use seaborn for grouped bars
    sns.barplot(
        data=df,
        x="Model",
        y="Accuracy",
        hue="Mode",
        ax=ax,
        palette=["#3498db", "#e74c3c"]
    )
    
    ax.set_xlabel('Model Variant', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Breakdown by Model and Mode', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(title='Mode')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved breakdown chart to: {output_path}")


def create_improvement_chart(
    all_results: List[ExperimentResult],
    output_path: str
):
    """Create chart showing improvement from plain to keyed mode."""
    
    # Group by model
    by_model = {}
    for r in all_results:
        if r.name not in by_model:
            by_model[r.name] = {}
        by_model[r.name][r.mode] = r.accuracy
    
    # Calculate improvements
    models = []
    improvements = []
    colors = []
    
    for model_name, modes in by_model.items():
        if "plain" in modes and "keyed" in modes:
            models.append(model_name)
            imp = modes["keyed"] - modes["plain"]
            improvements.append(imp)
            colors.append('#2ecc71' if imp > 0 else '#e74c3c')
    
    if not models:
        print("Cannot compute improvements - need both plain and keyed results")
        return
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, improvements, color=colors)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Model Variant', fontsize=12)
    ax.set_ylabel('Accuracy Change (Keyed - Plain)', fontsize=12)
    ax.set_title('Improvement from Anchor-Cycle Method', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(f'{val:+.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved improvement chart to: {output_path}")


def generate_summary_report(
    all_results: List[ExperimentResult],
    output_path: str
):
    """Generate a comprehensive markdown report."""
    
    lines = [
        "# FactKey Experiment Report",
        "",
        "## Overview",
        "",
        "This report compares the Baseline model (trained on plain forward facts) ",
        "with the Anchor-Cycle model (trained with anchor keys and KV cards) ",
        "on reverse factual queries.",
        "",
        "## Results Summary",
        "",
    ]
    
    # Create table
    lines.append("| Model | Mode | Accuracy | Correct | Total |")
    lines.append("|-------|------|----------|---------|-------|")
    
    for r in all_results:
        lines.append(f"| {r.name} | {r.mode} | {r.accuracy:.4f} | {r.correct} | {r.total} |")
    
    lines.append("")
    
    # Calculate improvements
    by_model = {}
    for r in all_results:
        if r.name not in by_model:
            by_model[r.name] = {}
        by_model[r.name][r.mode] = r
    
    lines.append("## Improvement Analysis")
    lines.append("")
    
    for model_name, modes in by_model.items():
        if "plain" in modes and "keyed" in modes:
            plain_acc = modes["plain"].accuracy
            keyed_acc = modes["keyed"].accuracy
            improvement = keyed_acc - plain_acc
            
            lines.append(f"### {model_name}")
            lines.append(f"- Plain mode accuracy: {plain_acc:.4f}")
            lines.append(f"- Keyed mode accuracy: {keyed_acc:.4f}")
            lines.append(f"- **Improvement: {improvement:+.4f}**")
            lines.append("")
    
    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    
    # Find baseline and anchor models
    baseline_keyed = None
    anchor_keyed = None
    for r in all_results:
        if "baseline" in r.name.lower() and r.mode == "keyed":
            baseline_keyed = r
        if "anchor" in r.name.lower() and r.mode == "keyed":
            anchor_keyed = r
    
    if baseline_keyed and anchor_keyed:
        diff = anchor_keyed.accuracy - baseline_keyed.accuracy
        lines.append(f"1. **Anchor-Cycle Method Effectiveness**: ")
        lines.append(f"   The anchor model with keyed queries achieves {anchor_keyed.accuracy:.4f} accuracy ")
        lines.append(f"   compared to {baseline_keyed.accuracy:.4f} for baseline with keyed queries.")
        lines.append(f"   This represents a **{diff:+.4f}** improvement.")
        lines.append("")
    
    lines.append("## Figures")
    lines.append("")
    lines.append("See the `outputs/figures/` directory for visualization charts:")
    lines.append("- `comparison_chart.png`: Bar chart comparing all configurations")
    lines.append("- `improvement_chart.png`: Improvement from plain to keyed mode")
    lines.append("- `breakdown_chart.png`: Detailed accuracy breakdown")
    lines.append("")
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Saved report to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare and visualize experiment results")
    
    parser.add_argument("--result_files", type=str, nargs="+", required=True,
                        help="Paths to evaluation result JSON files")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for figures and tables")
    parser.add_argument("--experiment_name", type=str, default="factkey_experiment",
                        help="Name for this experiment comparison")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("FactKey Results Comparison")
    print("="*60)
    
    # Create output directories
    figures_dir = Path(args.output_dir) / "figures"
    tables_dir = Path(args.output_dir) / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_results = []
    for result_file in args.result_files:
        print(f"Loading: {result_file}")
        results = load_eval_results(result_file)
        all_results.extend(results.values())
    
    print(f"Loaded {len(all_results)} result sets")
    
    # Generate outputs
    print("\nGenerating comparison outputs...")
    
    # 1. Comparison table (markdown)
    create_comparison_table(
        all_results,
        str(tables_dir / f"{args.experiment_name}_comparison.md"),
        format="markdown"
    )
    
    # 2. Comparison table (CSV)
    create_comparison_table(
        all_results,
        str(tables_dir / f"{args.experiment_name}_comparison.csv"),
        format="csv"
    )
    
    # 3. Comparison bar chart
    create_comparison_chart(
        all_results,
        str(figures_dir / f"{args.experiment_name}_comparison.png")
    )
    
    # 4. Improvement chart
    create_improvement_chart(
        all_results,
        str(figures_dir / f"{args.experiment_name}_improvement.png")
    )
    
    # 5. Breakdown chart
    create_detailed_breakdown_chart(
        all_results,
        str(figures_dir / f"{args.experiment_name}_breakdown.png")
    )
    
    # 6. Summary report
    generate_summary_report(
        all_results,
        str(tables_dir / f"{args.experiment_name}_report.md")
    )
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)
    print(f"Figures: {figures_dir}")
    print(f"Tables:  {tables_dir}")


if __name__ == "__main__":
    main()
