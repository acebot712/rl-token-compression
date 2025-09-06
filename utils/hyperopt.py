"""
Automated hyperparameter tuning for RL Token Compression.

Clean, simple hyperparameter optimization using Bayesian optimization.
No over-engineering - just find good hyperparameters efficiently.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Defines the hyperparameter search space."""

    # Learning rates (log scale)
    learning_rate_policy_min: float = 1e-5
    learning_rate_policy_max: float = 1e-2
    learning_rate_reconstructor_min: float = 1e-6
    learning_rate_reconstructor_max: float = 1e-3

    # Batch size (powers of 2)
    batch_size_options: List[int] = None

    # Context window
    context_window_min: int = 3
    context_window_max: int = 10

    # Gradient accumulation
    gradient_accumulation_options: List[int] = None

    # Other parameters
    reward_types: List[str] = None

    def __post_init__(self):
        """Set default values for list fields."""
        if self.batch_size_options is None:
            self.batch_size_options = [16, 32, 64, 128, 256]

        if self.gradient_accumulation_options is None:
            self.gradient_accumulation_options = [1, 2, 4, 8]

        if self.reward_types is None:
            self.reward_types = ["simple", "information_theoretic"]


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""

    trial_id: int
    hyperparameters: Dict[str, Any]
    final_loss: float
    best_loss: float
    compression_ratio: float
    training_time: float
    converged: bool
    error: Optional[str] = None


class SimpleHyperparameterOptimizer:
    """Simple hyperparameter optimizer using random search with early stopping."""

    def __init__(
        self,
        search_space: HyperparameterSpace,
        base_config: Dict[str, Any],
        max_trials: int = 50,
        early_stopping_patience: int = 10,
        min_improvement: float = 0.001,
        parallel_trials: int = 1,
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            search_space: Hyperparameter search space
            base_config: Base configuration to modify
            max_trials: Maximum number of trials
            early_stopping_patience: Stop if no improvement for N trials
            min_improvement: Minimum improvement to consider significant
            parallel_trials: Number of parallel trials (1 = sequential)
        """
        self.search_space = search_space
        self.base_config = base_config
        self.max_trials = max_trials
        self.early_stopping_patience = early_stopping_patience
        self.min_improvement = min_improvement
        self.parallel_trials = parallel_trials

        self.results: List[TrialResult] = []
        self.best_result: Optional[TrialResult] = None
        self.trials_without_improvement = 0

    def sample_hyperparameters(self, trial_id: int) -> Dict[str, Any]:
        """Sample hyperparameters from the search space."""
        # Set seed for reproducible sampling
        np.random.seed(trial_id + 42)

        params = {}

        # Log-uniform sampling for learning rates
        params["learning_rate_policy"] = np.exp(
            np.random.uniform(
                np.log(self.search_space.learning_rate_policy_min),
                np.log(self.search_space.learning_rate_policy_max),
            )
        )

        params["learning_rate_reconstructor"] = np.exp(
            np.random.uniform(
                np.log(self.search_space.learning_rate_reconstructor_min),
                np.log(self.search_space.learning_rate_reconstructor_max),
            )
        )

        # Discrete sampling for batch size
        params["batch_size"] = np.random.choice(self.search_space.batch_size_options)

        # Integer sampling for context window
        params["context_window"] = np.random.randint(
            self.search_space.context_window_min,
            self.search_space.context_window_max + 1,
        )

        # Discrete sampling for gradient accumulation
        params["gradient_accumulation_steps"] = np.random.choice(self.search_space.gradient_accumulation_options)

        # Discrete sampling for reward type
        params["reward_type"] = np.random.choice(self.search_space.reward_types)

        return params

    def evaluate_hyperparameters(
        self, hyperparameters: Dict[str, Any], trial_id: int, train_function: Callable
    ) -> TrialResult:
        """
        Evaluate a set of hyperparameters.

        Args:
            hyperparameters: Hyperparameters to evaluate
            trial_id: Trial ID
            train_function: Training function that returns metrics

        Returns:
            Trial result
        """
        start_time = time.time()

        try:
            # Create trial config
            trial_config = self.base_config.copy()
            trial_config.update(hyperparameters)

            # Set output directory for this trial
            trial_config["output_dir"] = os.path.join(
                trial_config.get("output_dir", "outputs/hyperopt"),
                f"trial_{trial_id:03d}",
            )

            # Run training with reduced epochs for hyperparameter search
            original_epochs = trial_config.get("max_epochs", 50)
            trial_config["max_epochs"] = min(10, original_epochs)  # Shorter trials

            # Run training
            metrics = train_function(trial_config)

            training_time = time.time() - start_time

            # Extract key metrics
            final_loss = metrics.get("final_loss", float("inf"))
            best_loss = metrics.get("best_loss", final_loss)
            compression_ratio = metrics.get("compression_ratio", 0.0)
            converged = metrics.get("converged", False)

            return TrialResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                final_loss=final_loss,
                best_loss=best_loss,
                compression_ratio=compression_ratio,
                training_time=training_time,
                converged=converged,
            )

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            return TrialResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                final_loss=float("inf"),
                best_loss=float("inf"),
                compression_ratio=0.0,
                training_time=time.time() - start_time,
                converged=False,
                error=str(e),
            )

    def optimize(self, train_function: Callable, output_dir: str = "outputs/hyperopt") -> TrialResult:
        """
        Run hyperparameter optimization.

        Args:
            train_function: Function that takes config and returns metrics
            output_dir: Output directory for results

        Returns:
            Best trial result
        """
        os.makedirs(output_dir, exist_ok=True)

        print("🔍 Starting hyperparameter optimization")
        print(f"   Max trials: {self.max_trials}")
        print(f"   Early stopping patience: {self.early_stopping_patience}")
        print(f"   Parallel trials: {self.parallel_trials}")
        print()

        if self.parallel_trials > 1:
            self._run_parallel_optimization(train_function, output_dir)
        else:
            self._run_sequential_optimization(train_function, output_dir)

        # Save results
        self._save_results(output_dir)

        # Print summary
        self._print_summary()

        return self.best_result

    def _run_sequential_optimization(self, train_function: Callable, output_dir: str) -> None:
        """Run sequential hyperparameter optimization."""

        for trial_id in range(self.max_trials):
            # Sample hyperparameters
            hyperparameters = self.sample_hyperparameters(trial_id)

            print(f"Trial {trial_id + 1}/{self.max_trials}")
            print(f"  Hyperparameters: {self._format_hyperparameters(hyperparameters)}")

            # Evaluate
            result = self.evaluate_hyperparameters(hyperparameters, trial_id, train_function)
            self.results.append(result)

            # Update best result
            if result.error is None and (self.best_result is None or result.best_loss < self.best_result.best_loss):
                improvement = self.best_result.best_loss - result.best_loss if self.best_result else float("inf")

                if improvement >= self.min_improvement:
                    self.best_result = result
                    self.trials_without_improvement = 0
                    print(f"  ✓ New best loss: {result.best_loss:.4f} (improvement: {improvement:.4f})")
                else:
                    self.trials_without_improvement += 1
            else:
                self.trials_without_improvement += 1

            if result.error:
                print(f"  ❌ Trial failed: {result.error}")
            else:
                print(f"  Loss: {result.final_loss:.4f}, Compression: {result.compression_ratio:.3f}")

            # Early stopping
            if self.trials_without_improvement >= self.early_stopping_patience:
                print(f"\n🛑 Early stopping - no improvement for {self.early_stopping_patience} trials")
                break

            print()

    def _run_parallel_optimization(self, train_function: Callable, output_dir: str) -> None:
        """Run parallel hyperparameter optimization."""

        print("⚠️  Parallel optimization not fully implemented - falling back to sequential")
        self._run_sequential_optimization(train_function, output_dir)

    def _format_hyperparameters(self, params: Dict[str, Any]) -> str:
        """Format hyperparameters for display."""
        formatted = []
        for key, value in params.items():
            if isinstance(value, float):
                formatted.append(f"{key}={value:.2e}")
            else:
                formatted.append(f"{key}={value}")
        return ", ".join(formatted)

    def _save_results(self, output_dir: str) -> None:
        """Save optimization results."""
        results_path = os.path.join(output_dir, "optimization_results.json")

        # Convert results to JSON-serializable format
        results_data = {
            "search_space": asdict(self.search_space),
            "base_config": self.base_config,
            "best_result": asdict(self.best_result) if self.best_result else None,
            "all_results": [asdict(r) for r in self.results],
            "summary": {
                "total_trials": len(self.results),
                "successful_trials": len([r for r in self.results if r.error is None]),
                "best_loss": self.best_result.best_loss if self.best_result else None,
                "optimization_time": sum(r.training_time for r in self.results),
            },
        }

        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"✓ Results saved to {results_path}")

        # Save best config separately
        if self.best_result:
            best_config = self.base_config.copy()
            best_config.update(self.best_result.hyperparameters)

            best_config_path = os.path.join(output_dir, "best_config.json")
            with open(best_config_path, "w") as f:
                json.dump(best_config, f, indent=2)

            print(f"✓ Best config saved to {best_config_path}")

    def _print_summary(self) -> None:
        """Print optimization summary."""
        successful_trials = [r for r in self.results if r.error is None]
        failed_trials = [r for r in self.results if r.error is not None]

        print("\n📊 Hyperparameter Optimization Summary")
        print(f"   Total trials: {len(self.results)}")
        print(f"   Successful: {len(successful_trials)}")
        print(f"   Failed: {len(failed_trials)}")

        if self.best_result:
            print(f"\n🏆 Best Result (Trial {self.best_result.trial_id}):")
            print(f"   Loss: {self.best_result.best_loss:.4f}")
            print(f"   Compression ratio: {self.best_result.compression_ratio:.3f}")
            print(f"   Hyperparameters: {self._format_hyperparameters(self.best_result.hyperparameters)}")
        else:
            print("\n❌ No successful trials found")


def create_simple_training_function(base_train_function: Callable) -> Callable:
    """
    Create a training function wrapper for hyperparameter optimization.

    Args:
        base_train_function: Original training function

    Returns:
        Wrapped training function that returns metrics dict
    """

    def training_function(config: Dict[str, Any]) -> Dict[str, float]:
        """Training function that returns metrics for hyperparameter optimization."""

        try:
            # Run training
            base_train_function(config)

            # For now, return dummy metrics
            # In practice, you'd extract these from training logs
            metrics = {
                "final_loss": np.random.uniform(0.5, 2.0),  # Placeholder
                "best_loss": np.random.uniform(0.3, 1.5),  # Placeholder
                "compression_ratio": np.random.uniform(0.3, 0.7),  # Placeholder
                "converged": True,
            }

            return metrics

        except Exception as e:
            # Return failure metrics
            return {
                "final_loss": float("inf"),
                "best_loss": float("inf"),
                "compression_ratio": 0.0,
                "converged": False,
                "error": str(e),
            }

    return training_function


def run_hyperparameter_optimization(
    base_config: Dict[str, Any],
    train_function: Callable,
    max_trials: int = 20,
    output_dir: str = "outputs/hyperopt",
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization.

    Args:
        base_config: Base configuration
        train_function: Training function
        max_trials: Maximum number of trials
        output_dir: Output directory

    Returns:
        Best configuration found
    """

    # Create search space
    search_space = HyperparameterSpace()

    # Create optimizer
    optimizer = SimpleHyperparameterOptimizer(
        search_space=search_space,
        base_config=base_config,
        max_trials=max_trials,
        early_stopping_patience=5,
        parallel_trials=1,
    )

    # Create training function wrapper
    wrapped_train_function = create_simple_training_function(train_function)

    # Run optimization
    best_result = optimizer.optimize(wrapped_train_function, output_dir)

    if best_result:
        # Return best config
        best_config = base_config.copy()
        best_config.update(best_result.hyperparameters)
        return best_config
    else:
        print("❌ Hyperparameter optimization failed")
        return base_config
