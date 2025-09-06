"""
Tests for configuration schema validation.
"""

import pytest

from utils.schema import (
    ConfigValidator,
    ValidationError,
    validate_config,
    validate_data_config,
    validate_evaluation_config,
    validate_training_config,
)


class TestConfigValidator:
    """Test the base ConfigValidator methods."""

    def test_validate_positive_number_valid(self):
        """Test positive number validation with valid inputs."""
        ConfigValidator.validate_positive_number(5, "test_field")
        ConfigValidator.validate_positive_number(0.1, "test_field")
        ConfigValidator.validate_positive_number(1000, "test_field")

    def test_validate_positive_number_invalid(self):
        """Test positive number validation with invalid inputs."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_positive_number(-1, "test_field")
        assert "must be positive" in str(exc_info.value)

        with pytest.raises(ValidationError):
            ConfigValidator.validate_positive_number("not_a_number", "test_field")

    def test_validate_string_choice_valid(self):
        """Test string choice validation with valid inputs."""
        choices = ["option1", "option2", "option3"]
        ConfigValidator.validate_string_choice("option1", "test_field", choices)
        ConfigValidator.validate_string_choice("option3", "test_field", choices)

    def test_validate_string_choice_invalid(self):
        """Test string choice validation with invalid inputs."""
        choices = ["option1", "option2"]

        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_string_choice("invalid_option", "test_field", choices)
        assert "must be one of" in str(exc_info.value)

        with pytest.raises(ValidationError):
            ConfigValidator.validate_string_choice(123, "test_field", choices)

    def test_validate_range_valid(self):
        """Test range validation with valid inputs."""
        ConfigValidator.validate_range(0.5, "test_field", 0.0, 1.0)
        ConfigValidator.validate_range(10, "test_field", 5, 15)

    def test_validate_range_invalid(self):
        """Test range validation with invalid inputs."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_range(1.5, "test_field", 0.0, 1.0)
        assert "must be between" in str(exc_info.value)

        with pytest.raises(ValidationError):
            ConfigValidator.validate_range("not_a_number", "test_field", 0, 10)


class TestDataConfigValidation:
    """Test data configuration validation."""

    def test_valid_data_config(self):
        """Test validation of valid data config."""
        config = {
            "output_dir": "data/processed",
            "max_sequences": 1000,
            "max_length": 512,
            "min_length": 10,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "dataset_name": "reddit",
            "tokenizer": "gpt2",
        }

        # Should not raise any exception
        validate_data_config(config)

    def test_missing_required_fields(self):
        """Test validation fails with missing required fields."""
        config = {
            "max_sequences": 1000
            # Missing output_dir, max_length
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_data_config(config)
        assert "output_dir" in str(exc_info.value) or "max_length" in str(exc_info.value)

    def test_invalid_splits(self):
        """Test validation fails with invalid split ratios."""
        config = {
            "output_dir": "data/processed",
            "max_sequences": 1000,
            "max_length": 512,
            "train_split": 0.9,
            "val_split": 0.2,
            "test_split": 0.1,
            # Splits sum to 1.2, not 1.0
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_data_config(config)
        assert "must equal 1.0" in str(exc_info.value)

    def test_invalid_dataset_name(self):
        """Test validation fails with invalid dataset name."""
        config = {
            "output_dir": "data/processed",
            "max_sequences": 1000,
            "max_length": 512,
            "dataset_name": "invalid_dataset",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_data_config(config)
        assert "dataset_name" in str(exc_info.value)


class TestTrainingConfigValidation:
    """Test training configuration validation."""

    def test_valid_training_config(self):
        """Test validation of valid training config."""
        config = {
            "data_path": "data/train.json",
            "reconstructor_path": "models/reconstructor",
            "output_dir": "outputs/training",
            "learning_rate_policy": 0.001,
            "learning_rate_reconstructor": 0.0001,
            "batch_size": 32,
            "max_epochs": 100,
            "context_window": 5,
            "device": "auto",
            "reward_type": "simple",
        }

        # Should not raise any exception
        validate_training_config(config)

    def test_invalid_learning_rate(self):
        """Test validation fails with invalid learning rate."""
        config = {
            "data_path": "data/train.json",
            "reconstructor_path": "models/reconstructor",
            "output_dir": "outputs/training",
            "learning_rate_policy": 2.0,  # Too high
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_training_config(config)
        assert "learning_rate_policy" in str(exc_info.value)

    def test_large_batch_size_warning(self):
        """Test validation fails with excessively large batch size."""
        config = {
            "data_path": "data/train.json",
            "reconstructor_path": "models/reconstructor",
            "output_dir": "outputs/training",
            "batch_size": 1024,  # Very large
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_training_config(config)
        assert "memory issues" in str(exc_info.value)

    def test_micro_batch_consistency(self):
        """Test micro batch size consistency with batch size."""
        config = {
            "data_path": "data/train.json",
            "reconstructor_path": "models/reconstructor",
            "output_dir": "outputs/training",
            "batch_size": 32,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 2,
            # micro_batch * grad_accum = 16, but batch_size = 32
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_training_config(config)
        assert "must equal batch_size" in str(exc_info.value)


class TestEvaluationConfigValidation:
    """Test evaluation configuration validation."""

    def test_valid_evaluation_config(self):
        """Test validation of valid evaluation config."""
        config = {
            "data_path": "data/test.json",
            "output_dir": "outputs/evaluation",
            "num_sequences": 1000,
            "baselines": ["random", "frequency"],
            "metrics": ["compression_ratio", "bleu_score"],
            "target_ratios": [0.3, 0.5, 0.7],
            "device": "auto",
        }

        # Should not raise any exception
        validate_evaluation_config(config)

    def test_invalid_baseline(self):
        """Test validation fails with invalid baseline."""
        config = {
            "data_path": "data/test.json",
            "output_dir": "outputs/evaluation",
            "baselines": ["random", "invalid_baseline"],
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_evaluation_config(config)
        assert "baselines" in str(exc_info.value)

    def test_invalid_target_ratios(self):
        """Test validation fails with invalid target ratios."""
        config = {
            "data_path": "data/test.json",
            "output_dir": "outputs/evaluation",
            "target_ratios": [0.3, 1.5, 0.7],  # 1.5 is > 1.0
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_evaluation_config(config)
        assert "target_ratios" in str(exc_info.value)


class TestGeneralValidation:
    """Test general validation functions."""

    def test_validate_config_dispatcher(self):
        """Test that validate_config dispatches to correct validator."""
        config = {
            "output_dir": "data/processed",
            "max_sequences": 1000,
            "max_length": 512,
        }

        # Should work for data config
        validate_config(config, "data")

        # Should fail for training config (missing required fields)
        with pytest.raises(ValidationError):
            validate_config(config, "training")

    def test_unknown_config_type(self):
        """Test validation fails with unknown config type."""
        config = {"test": "value"}

        with pytest.raises(ValueError) as exc_info:
            validate_config(config, "unknown_type")
        assert "Unknown config type" in str(exc_info.value)

    def test_validation_error_formatting(self):
        """Test that ValidationError formats correctly."""
        error = ValidationError("test_field", 42, "test message")

        error_str = str(error)
        assert "test_field" in error_str
        assert "test message" in error_str
        assert "42" in error_str
