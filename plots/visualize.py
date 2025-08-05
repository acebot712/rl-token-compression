"""
Comprehensive visualization utilities for token compression evaluation.

This module provides the main visualization functions expected by the evaluation
pipeline, leveraging the plotting utilities from src.utils.plotting.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add project root to path to import our utilities
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.plotting import (
    create_research_dashboard,
    create_comparison_bar_plot,
    create_training_curves,
    create_scatter_plot,
    setup_plot_style
)

logger = logging.getLogger(__name__)


def create_comprehensive_visualization(
    training_data: Optional[Dict[str, List[float]]] = None,
    baseline_data: Optional[Dict[str, float]] = None,
    compression_quality_data: Optional[Tuple[List[float], List[float]]] = None,
    output_dir: str = "plots",
    file_prefix: str = "evaluation"
) -> Dict[str, str]:
    """
    Create comprehensive visualization suite for token compression evaluation.
    
    This is the main function called by the evaluation pipeline to generate
    all necessary plots and visualizations.
    
    Args:
        training_data: Dictionary with training metrics (e.g., {'policy_loss': [...], 'reward': [...]})
        baseline_data: Dictionary with baseline method performance scores
        compression_quality_data: Tuple of (compression_ratios, quality_scores) for trade-off analysis
        output_dir: Directory to save plots
        file_prefix: Prefix for generated files
        
    Returns:
        Dictionary mapping plot names to file paths of generated visualizations
    """
    logger.info(f"Creating comprehensive visualizations in {output_dir}")
    
    # Set up plotting style
    setup_plot_style()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    generated_plots = {}
    
    # Import matplotlib at the beginning to avoid scoping issues
    import matplotlib.pyplot as plt
    
    try:
        # 1. Research Dashboard (if we have all data)
        if training_data and baseline_data and compression_quality_data:
            logger.info("Creating research dashboard...")
            dashboard_fig = create_research_dashboard(
                baseline_data=baseline_data,
                training_data=training_data,
                compression_quality_data=compression_quality_data,
                output_dir=output_dir,
                filename=f"{file_prefix}_dashboard"
            )
            generated_plots['dashboard'] = os.path.join(output_dir, f"{file_prefix}_dashboard.png")
            plt.close(dashboard_fig)  # Free memory
        
        # 2. Baseline Comparison Plot
        if baseline_data:
            logger.info("Creating baseline comparison plot...")
            baseline_fig = create_comparison_bar_plot(
                data=baseline_data,
                title="Baseline Method Comparison",
                ylabel="Performance Score",
                output_dir=output_dir,
                filename=f"{file_prefix}_baseline_comparison"
            )
            generated_plots['baseline_comparison'] = os.path.join(output_dir, f"{file_prefix}_baseline_comparison.png")
            plt.close(baseline_fig)  # Free memory
        
        # 3. Training Progress Curves
        if training_data:
            logger.info("Creating training curves...")
            training_fig = create_training_curves(
                training_data=training_data,
                title="Training Progress",
                output_dir=output_dir,
                filename=f"{file_prefix}_training_curves"
            )
            generated_plots['training_curves'] = os.path.join(output_dir, f"{file_prefix}_training_curves.png")
            plt.close(training_fig)  # Free memory
        
        # 4. Compression vs Quality Trade-off
        if compression_quality_data:
            logger.info("Creating compression-quality trade-off plot...")
            compression_ratios, quality_scores = compression_quality_data
            
            # Create labels for the scatter plot (if we have baseline data)
            labels = None
            if baseline_data and len(compression_ratios) == len(baseline_data):
                labels = list(baseline_data.keys())
            
            scatter_fig = create_scatter_plot(
                x_data=compression_ratios,
                y_data=quality_scores,
                labels=labels,
                title="Compression vs Quality Trade-off",
                xlabel="Compression Ratio",
                ylabel="Quality Score (BLEU)",
                output_dir=output_dir,
                filename=f"{file_prefix}_compression_quality"
            )
            generated_plots['compression_quality'] = os.path.join(output_dir, f"{file_prefix}_compression_quality.png")
            plt.close(scatter_fig)  # Free memory
        
        # 5. Individual Baseline Performance (detailed view)
        if baseline_data:
            logger.info("Creating detailed baseline performance plot...")
            
            # Create a more detailed view with error bars if available
            detailed_fig = create_comparison_bar_plot(
                data=baseline_data,
                title="Detailed Baseline Performance Analysis",
                ylabel="Compression Quality (BLEU Score)",
                output_dir=output_dir,
                filename=f"{file_prefix}_detailed_baselines"
            )
            generated_plots['detailed_baselines'] = os.path.join(output_dir, f"{file_prefix}_detailed_baselines.png")
            plt.close(detailed_fig)  # Free memory
        
        logger.info(f"Successfully created {len(generated_plots)} visualization plots")
        
        # Log the created files
        for plot_name, file_path in generated_plots.items():
            if os.path.exists(file_path):
                logger.info(f"  ✓ {plot_name}: {file_path}")
            else:
                logger.warning(f"  ✗ {plot_name}: {file_path} (file not found)")
        
        return generated_plots
        
    except Exception as e:
        logger.error(f"Failed to create comprehensive visualizations: {e}")
        return {'error': str(e)}


def create_evaluation_summary_plot(
    baseline_results: Dict[str, Any],
    model_results: Optional[Dict[str, Any]] = None,
    output_dir: str = "plots",
    filename: str = "evaluation_summary"
) -> Optional[str]:
    """
    Create a summary plot specifically for evaluation results.
    
    Args:
        baseline_results: Results from baseline evaluation
        model_results: Results from model evaluation (optional)
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to generated plot file, or None if failed
    """
    try:
        # Extract performance scores from baseline results
        performance_data = {}
        
        for method_name, method_data in baseline_results.items():
            if "error" not in method_data:
                # Extract first ratio's reconstruction quality as performance metric
                first_ratio_key = next((k for k in method_data.keys() if k.startswith('ratio_')), None)
                if first_ratio_key and isinstance(method_data[first_ratio_key], dict):
                    ratio_data = method_data[first_ratio_key]
                    if 'reconstruction_quality' in ratio_data and isinstance(ratio_data['reconstruction_quality'], dict):
                        performance_data[method_name] = ratio_data['reconstruction_quality']['mean']
        
        # Add model results if available
        if model_results and 'quality_metrics' in model_results:
            # Extract model performance
            first_ratio = next(iter(model_results['quality_metrics'].values()))
            if isinstance(first_ratio, dict) and 'mean' in first_ratio:
                performance_data['rl_model'] = first_ratio['mean']
        
        if performance_data:
            fig = create_comparison_bar_plot(
                data=performance_data,
                title="Evaluation Summary: Method Performance Comparison",
                ylabel="Reconstruction Quality (BLEU Score)",
                output_dir=output_dir,
                filename=filename
            )
            output_path = os.path.join(output_dir, f"{filename}.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
            return output_path
        else:
            logger.warning("No performance data available for summary plot")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create evaluation summary plot: {e}")
        return None


# Convenience function for backward compatibility
def create_plots(
    training_data: Optional[Dict[str, List[float]]] = None,
    baseline_data: Optional[Dict[str, float]] = None,
    output_dir: str = "plots"
) -> Dict[str, str]:
    """
    Simplified plotting function for basic use cases.
    
    Args:
        training_data: Training progress data
        baseline_data: Baseline comparison data
        output_dir: Output directory
        
    Returns:
        Dictionary of generated plot files
    """
    return create_comprehensive_visualization(
        training_data=training_data,
        baseline_data=baseline_data,
        output_dir=output_dir,
        file_prefix="basic"
    )