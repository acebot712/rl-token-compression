#!/usr/bin/env python3
"""
Demo script to test the comprehensive visualization system.

This script demonstrates the research-quality visualizations that will be
generated during evaluation, including baseline comparisons, training curves,
and compression-quality trade-off analysis.
"""

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from plots.visualize import create_comprehensive_visualization

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_visualizations():
    """Create comprehensive demo visualizations."""
    logger.info("Creating demo visualizations for token compression research")
    
    # Demo training data (simulated joint training progress)
    training_data = {
        'policy_loss': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15],
        'reconstructor_loss': [2.5, 2.3, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1],
        'reward': [0.1, 0.2, 0.35, 0.45, 0.55, 0.62, 0.68, 0.72, 0.75, 0.78],
        'compression_ratio': [0.2, 0.25, 0.3, 0.35, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52],
        'temperature': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    }
    
    # Demo baseline comparison data
    baseline_data = {
        'random': 0.25,        # Random masking baseline
        'frequency': 0.45,     # Frequency-based masking
        'length': 0.35,        # Length-based masking  
        'position': 0.40,      # Position-based masking
        'entropy': 0.50,       # Entropy-based masking
        'rl_model': 0.78       # Our RL model (best performer)
    }
    
    # Demo compression vs quality trade-off data
    compression_quality_data = (
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],     # compression ratios
        [0.9, 0.85, 0.78, 0.7, 0.6, 0.45, 0.3]   # quality scores (BLEU/ROUGE)
    )
    
    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, "demo_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive visualizations
    figures = create_comprehensive_visualization(
        training_data=training_data,
        baseline_data=baseline_data,
        compression_quality_data=compression_quality_data,
        output_dir=output_dir
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("DEMO VISUALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Generated {len(figures)} research-quality plots:")
    
    for plot_name in figures.keys():
        logger.info(f"  ✓ {plot_name}")
    
    logger.info(f"\nPlots saved to: {output_dir}")
    logger.info("\nGenerated plots include:")
    logger.info("  • Research dashboard with all key metrics")
    logger.info("  • Training progress curves")
    logger.info("  • Baseline method comparison")
    logger.info("  • Compression vs quality trade-off analysis")
    
    logger.info("\nThese visualizations demonstrate:")
    logger.info("  • RL model significantly outperforms baselines")
    logger.info("  • Joint training converges successfully")
    logger.info("  • Clear compression-quality trade-off")
    logger.info("  • Publication-ready figure quality")
    
    return figures


if __name__ == "__main__":
    try:
        figures = create_demo_visualizations()
        print(f"\n✅ Success! Generated {len(figures)} visualization plots.")
        print("Check demo_plots/ directory for the results.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure matplotlib and seaborn are installed:")
        print("  pip install matplotlib seaborn")
        sys.exit(1)