"""
Shared plotting utilities for token compression research.

Provides consistent styling, error handling, and export functions
for all visualization components.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Publication-quality settings
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Professional color schemes
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'baseline': '#7C7C7C',     # Gray
    'rl_model': '#2E86AB',     # Blue
    'best': '#0F7B0F',         # Green
}

BASELINE_COLORS = {
    'random': '#7C7C7C',
    'frequency': '#A23B72', 
    'length': '#F18F01',
    'position': '#C73E1D',
    'entropy': '#2E86AB',
    'rl_model': '#0F7B0F'
}


def setup_plot_style():
    """Set up consistent plot styling."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.style.use('seaborn-v0_8-darkgrid')


def save_plot(fig, output_dir: str, filename: str, formats: List[str] = None):
    """
    Save plot in multiple formats with proper error handling.
    
    Args:
        fig: Matplotlib figure
        output_dir: Output directory
        filename: Base filename (without extension)
        formats: List of formats to save ['png', 'pdf', 'svg']
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for fmt in formats:
        try:
            output_path = os.path.join(output_dir, f"{filename}.{fmt}")
            fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save plot in {fmt} format: {e}")


def add_significance_stars(ax, x1: float, x2: float, y: float, p_value: float):
    """Add significance stars above bars."""
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'ns'
    
    # Draw line
    ax.plot([x1, x2], [y, y], 'k-', lw=1)
    # Add stars
    ax.text((x1 + x2) / 2, y + 0.01, stars, ha='center', va='bottom')


def create_comparison_bar_plot(
    data: Dict[str, float],
    errors: Dict[str, float] = None,
    title: str = "Performance Comparison",
    ylabel: str = "Score",
    output_dir: str = None,
    filename: str = "comparison"
) -> plt.Figure:
    """
    Create a professional bar plot for baseline comparisons.
    
    Args:
        data: Dictionary of method names to values
        errors: Dictionary of method names to error values
        title: Plot title
        ylabel: Y-axis label
        output_dir: Output directory (saves if provided)
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = list(data.keys())
    values = list(data.values())
    error_vals = [errors.get(method, 0) for method in methods] if errors else None
    
    # Color bars based on method type
    colors = [BASELINE_COLORS.get(method, COLORS['baseline']) for method in methods]
    
    bars = ax.bar(methods, values, yerr=error_vals, capsize=5, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Highlight best performer
    best_idx = np.argmax(values)
    bars[best_idx].set_color(COLORS['best'])
    bars[best_idx].set_alpha(1.0)
    
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (error_vals[bars.index(bar)] if error_vals else 0),
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        save_plot(fig, output_dir, filename)
    
    return fig


def create_training_curves(
    training_data: Dict[str, List[float]],
    title: str = "Training Progress",
    output_dir: str = None,
    filename: str = "training_curves"
) -> plt.Figure:
    """
    Create training progress curves.
    
    Args:
        training_data: Dictionary with keys like 'policy_loss', 'reconstructor_loss', 'reward'
        title: Plot title
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    n_plots = len(training_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (metric, values) in enumerate(training_data.items()):
        ax = axes[i]
        steps = list(range(len(values)))
        
        ax.plot(steps, values, color=COLORS['primary'], linewidth=2, alpha=0.8)
        
        # Add trend line
        if len(values) > 10:
            z = np.polyfit(steps, values, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), "--", color=COLORS['accent'], alpha=0.8, linewidth=1)
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        save_plot(fig, output_dir, filename)
    
    return fig


def create_scatter_plot(
    x_data: List[float],
    y_data: List[float],
    labels: List[str] = None,
    title: str = "Scatter Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    output_dir: str = None,
    filename: str = "scatter"
) -> plt.Figure:
    """
    Create scatter plot with trend line.
    
    Args:
        x_data: X coordinates
        y_data: Y coordinates  
        labels: Point labels
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    colors = [BASELINE_COLORS.get(label, COLORS['baseline']) if labels else COLORS['primary'] 
              for label in (labels or [''] * len(x_data))]
    
    scatter = ax.scatter(x_data, y_data, c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    
    # Add labels if provided
    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (x_data[i], y_data[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
    
    # Add trend line
    if len(x_data) > 2:
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(x_data), max(x_data), 100)
        ax.plot(x_trend, p(x_trend), '--', color=COLORS['accent'], alpha=0.8, linewidth=2)
        
        # Add correlation coefficient
        corr = np.corrcoef(x_data, y_data)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        save_plot(fig, output_dir, filename)
    
    return fig


def create_heatmap(
    data: np.ndarray,
    x_labels: List[str] = None,
    y_labels: List[str] = None,
    title: str = "Heatmap",
    output_dir: str = None,
    filename: str = "heatmap"
) -> plt.Figure:
    """
    Create heatmap visualization.
    
    Args:
        data: 2D numpy array
        x_labels: X-axis labels
        y_labels: Y-axis labels
        title: Plot title
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Set labels
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    ax.set_title(title, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_dir:
        save_plot(fig, output_dir, filename)
    
    return fig


def create_research_dashboard(
    baseline_data: Dict[str, float],
    training_data: Dict[str, List[float]],
    compression_quality_data: Tuple[List[float], List[float]],
    output_dir: str,
    filename: str = "research_dashboard"
) -> plt.Figure:
    """
    Create comprehensive research dashboard with multiple panels.
    
    Args:
        baseline_data: Baseline comparison data
        training_data: Training progress data  
        compression_quality_data: (compression_ratios, quality_scores) tuple
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Baseline comparison
    ax1 = plt.subplot(2, 2, 1)
    methods = list(baseline_data.keys())
    values = list(baseline_data.values())
    colors = [BASELINE_COLORS.get(method, COLORS['baseline']) for method in methods]
    
    bars = ax1.bar(methods, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Baseline Comparison', fontweight='bold')
    ax1.set_ylabel('Performance Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Panel 2: Training curves
    ax2 = plt.subplot(2, 2, 2)
    for metric, data in training_data.items():
        steps = list(range(len(data)))
        ax2.plot(steps, data, label=metric.replace('_', ' ').title(), linewidth=2)
    ax2.set_title('Training Progress', fontweight='bold')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Compression vs Quality
    ax3 = plt.subplot(2, 2, 3)
    compression_ratios, quality_scores = compression_quality_data
    ax3.scatter(compression_ratios, quality_scores, alpha=0.7, s=50, 
               color=COLORS['primary'], edgecolors='black', linewidth=0.5)
    
    # Add trend line
    if len(compression_ratios) > 2:
        z = np.polyfit(compression_ratios, quality_scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(compression_ratios), max(compression_ratios), 100)
        ax3.plot(x_trend, p(x_trend), '--', color=COLORS['accent'], alpha=0.8, linewidth=2)
    
    ax3.set_title('Compression vs Quality Trade-off', fontweight='bold')
    ax3.set_xlabel('Compression Ratio')
    ax3.set_ylabel('Quality Score')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Summary statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate summary stats
    best_baseline = max(baseline_data, key=baseline_data.get)
    best_score = baseline_data[best_baseline]
    avg_compression = np.mean(compression_ratios) if compression_ratios else 0
    avg_quality = np.mean(quality_scores) if quality_scores else 0
    
    summary_text = f"""
    Summary Statistics
    
    Best Baseline: {best_baseline}
    Best Score: {best_score:.3f}
    
    Avg Compression: {avg_compression:.3f}
    Avg Quality: {avg_quality:.3f}
    
    Training Steps: {len(list(training_data.values())[0]) if training_data else 0}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Token Compression Research Dashboard', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        save_plot(fig, output_dir, filename)
    
    return fig