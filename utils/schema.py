"""
Configuration schema validation.

Validates configurations to prevent runtime errors from invalid parameters.
"""

from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
import json


@dataclass
class ValidationError(Exception):
    """Configuration validation error."""
    field: str
    value: Any
    message: str
    
    def __str__(self):
        return f"Config validation error for '{self.field}': {self.message} (got: {self.value})"


class ConfigValidator:
    """Validates configuration dictionaries against schemas."""
    
    @staticmethod
    def validate_positive_number(value: Any, field: str, allow_zero: bool = False) -> None:
        """Validate that value is a positive number."""
        if not isinstance(value, (int, float)):
            raise ValidationError(field, value, "must be a number")
        
        if allow_zero and value < 0:
            raise ValidationError(field, value, "must be non-negative")
        elif not allow_zero and value <= 0:
            raise ValidationError(field, value, "must be positive")
    
    @staticmethod
    def validate_string_choice(value: Any, field: str, choices: List[str]) -> None:
        """Validate that value is one of the allowed choices."""
        if not isinstance(value, str):
            raise ValidationError(field, value, "must be a string")
        
        if value not in choices:
            raise ValidationError(field, value, f"must be one of {choices}")
    
    @staticmethod
    def validate_range(value: Any, field: str, min_val: float, max_val: float) -> None:
        """Validate that value is within specified range."""
        if not isinstance(value, (int, float)):
            raise ValidationError(field, value, "must be a number")
        
        if not (min_val <= value <= max_val):
            raise ValidationError(field, value, f"must be between {min_val} and {max_val}")
    
    @staticmethod
    def validate_list_of_numbers(value: Any, field: str) -> None:
        """Validate that value is a list of numbers."""
        if not isinstance(value, list):
            raise ValidationError(field, value, "must be a list")
        
        for i, item in enumerate(value):
            if not isinstance(item, (int, float)):
                raise ValidationError(f"{field}[{i}]", item, "must be a number")


def validate_data_config(config: Dict[str, Any]) -> None:
    """Validate data preparation configuration."""
    required_fields = ["output_dir", "max_sequences", "max_length"]
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValidationError(field, None, "is required")
    
    # Validate specific fields
    ConfigValidator.validate_positive_number(config["max_sequences"], "max_sequences")
    ConfigValidator.validate_positive_number(config["max_length"], "max_length")
    
    if "min_length" in config:
        ConfigValidator.validate_positive_number(config["min_length"], "min_length")
        if config["min_length"] >= config["max_length"]:
            raise ValidationError("min_length", config["min_length"], 
                                "must be less than max_length")
    
    if "train_split" in config:
        ConfigValidator.validate_range(config["train_split"], "train_split", 0.0, 1.0)
    
    if "val_split" in config:
        ConfigValidator.validate_range(config["val_split"], "val_split", 0.0, 1.0)
    
    if "test_split" in config:
        ConfigValidator.validate_range(config["test_split"], "test_split", 0.0, 1.0)
    
    # Validate split totals
    total_split = (config.get("train_split", 0.8) + 
                   config.get("val_split", 0.1) + 
                   config.get("test_split", 0.1))
    if abs(total_split - 1.0) > 0.01:
        raise ValidationError("splits", total_split, 
                            "train_split + val_split + test_split must equal 1.0")
    
    if "dataset_name" in config:
        allowed_datasets = ["reddit", "wikitext", "bookcorpus", "openwebtext"]
        ConfigValidator.validate_string_choice(config["dataset_name"], "dataset_name", 
                                             allowed_datasets)
    
    if "tokenizer" in config:
        allowed_tokenizers = ["gpt2", "gpt2-medium", "gpt2-large"]
        ConfigValidator.validate_string_choice(config["tokenizer"], "tokenizer", 
                                             allowed_tokenizers)


def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training configuration."""
    required_fields = ["data_path", "reconstructor_path", "output_dir"]
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValidationError(field, None, "is required")
    
    # Validate learning rates
    if "learning_rate_policy" in config:
        ConfigValidator.validate_range(config["learning_rate_policy"], 
                                     "learning_rate_policy", 1e-6, 1.0)
    
    if "learning_rate_reconstructor" in config:
        ConfigValidator.validate_range(config["learning_rate_reconstructor"], 
                                     "learning_rate_reconstructor", 1e-6, 1.0)
    
    # Validate training parameters
    if "batch_size" in config:
        ConfigValidator.validate_positive_number(config["batch_size"], "batch_size")
        if config["batch_size"] > 512:
            raise ValidationError("batch_size", config["batch_size"], 
                                "batch size > 512 may cause memory issues")
    
    if "max_epochs" in config:
        ConfigValidator.validate_positive_number(config["max_epochs"], "max_epochs")
    
    if "context_window" in config:
        ConfigValidator.validate_positive_number(config["context_window"], "context_window")
        if config["context_window"] > 20:
            raise ValidationError("context_window", config["context_window"], 
                                "context window > 20 may be ineffective")
    
    # Validate device
    if "device" in config:
        allowed_devices = ["auto", "cpu", "cuda", "mps"]
        ConfigValidator.validate_string_choice(config["device"], "device", allowed_devices)
    
    # Validate reward type
    if "reward_type" in config:
        allowed_rewards = ["simple", "information_theoretic"]
        ConfigValidator.validate_string_choice(config["reward_type"], "reward_type", 
                                             allowed_rewards)
    
    # Validate gradient accumulation
    if "gradient_accumulation_steps" in config:
        ConfigValidator.validate_positive_number(config["gradient_accumulation_steps"], 
                                                "gradient_accumulation_steps")
    
    if "micro_batch_size" in config:
        ConfigValidator.validate_positive_number(config["micro_batch_size"], 
                                                "micro_batch_size")
        
        # Check consistency with batch_size
        if "batch_size" in config:
            batch_size = config["batch_size"]
            micro_batch = config["micro_batch_size"]
            grad_accum = config.get("gradient_accumulation_steps", 1)
            
            expected_batch = micro_batch * grad_accum
            if expected_batch != batch_size:
                raise ValidationError("micro_batch_size", micro_batch,
                                    f"micro_batch_size * gradient_accumulation_steps "
                                    f"({expected_batch}) must equal batch_size ({batch_size})")


def validate_evaluation_config(config: Dict[str, Any]) -> None:
    """Validate evaluation configuration."""
    required_fields = ["data_path", "output_dir"]
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValidationError(field, None, "is required")
    
    # Validate number of sequences
    if "num_sequences" in config:
        ConfigValidator.validate_positive_number(config["num_sequences"], "num_sequences")
    
    # Validate baselines
    if "baselines" in config:
        allowed_baselines = ["random", "frequency", "length", "position", "learned"]
        if not isinstance(config["baselines"], list):
            raise ValidationError("baselines", config["baselines"], "must be a list")
        
        for i, baseline in enumerate(config["baselines"]):
            ConfigValidator.validate_string_choice(baseline, f"baselines[{i}]", 
                                                 allowed_baselines)
    
    # Validate metrics
    if "metrics" in config:
        allowed_metrics = ["compression_ratio", "bleu_score", "rouge_score", 
                          "reconstruction_loss", "perplexity"]
        if not isinstance(config["metrics"], list):
            raise ValidationError("metrics", config["metrics"], "must be a list")
        
        for i, metric in enumerate(config["metrics"]):
            ConfigValidator.validate_string_choice(metric, f"metrics[{i}]", 
                                                 allowed_metrics)
    
    # Validate target ratios
    if "target_ratios" in config:
        ConfigValidator.validate_list_of_numbers(config["target_ratios"], "target_ratios")
        for i, ratio in enumerate(config["target_ratios"]):
            ConfigValidator.validate_range(ratio, f"target_ratios[{i}]", 0.0, 1.0)
    
    if "device" in config:
        allowed_devices = ["auto", "cpu", "cuda", "mps"]
        ConfigValidator.validate_string_choice(config["device"], "device", allowed_devices)


def validate_config(config: Dict[str, Any], config_type: str) -> None:
    """Validate configuration based on type.
    
    Args:
        config: Configuration dictionary
        config_type: Type of configuration ("data", "training", "evaluation")
    """
    validators = {
        "data": validate_data_config,
        "training": validate_training_config,
        "evaluation": validate_evaluation_config
    }
    
    if config_type not in validators:
        raise ValueError(f"Unknown config type: {config_type}. "
                        f"Must be one of {list(validators.keys())}")
    
    try:
        validators[config_type](config)
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError("unknown", None, f"Unexpected validation error: {e}")


def validate_config_file(config_path: str, config_type: str) -> Dict[str, Any]:
    """Load and validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        config_type: Type of configuration
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file has invalid JSON
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    validate_config(config, config_type)
    return config