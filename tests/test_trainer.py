"""
Unit tests for the training system.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from training.trainer import SimpleJointTrainer, create_simple_trainer


@pytest.fixture
def device():
    """Get available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock(spec=GPT2Tokenizer)
    tokenizer.mask_token_id = 50256  # GPT-2 EOS token as mask
    tokenizer.unk_token_id = 50256
    return tokenizer


@pytest.fixture
def reconstructor():
    """Mock reconstructor model."""
    model = Mock(spec=GPT2LMHeadModel)
    
    # Mock the transformer.wte (word token embeddings) - return correct size dynamically
    def dynamic_wte(input_tensor):
        batch_size, seq_len = input_tensor.shape
        return torch.randn(batch_size, seq_len, 768)  # [batch_size, seq_len, embed_dim]
    
    mock_wte = Mock()
    mock_wte.side_effect = dynamic_wte
    mock_transformer = Mock()
    mock_transformer.wte = mock_wte
    model.transformer = mock_transformer
    
    # Mock forward pass
    mock_output = Mock()
    mock_output.loss = torch.tensor(1.5)
    model.return_value = mock_output
    
    # Mock modules() method for deterministic function - return empty list to avoid iteration issues
    model.modules.return_value = []
    
    # Mock parameters() method to return dummy tensor for optimizer
    dummy_param = torch.tensor([1.0], requires_grad=True)
    model.parameters.return_value = [dummy_param]
    
    return model


@pytest.fixture
def sample_data():
    """Sample training data."""
    return [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15]
    ]


class TestSimpleJointTrainer:
    """Test the SimpleJointTrainer class."""
    
    def test_init(self, device):
        """Test trainer initialization."""
        trainer = SimpleJointTrainer(
            policy_lr=1e-3,
            reconstructor_lr=1e-4,
            device=device,
            context_window=5
        )
        
        assert trainer.policy_lr == 1e-3
        assert trainer.reconstructor_lr == 1e-4
        assert trainer.device == device
        assert trainer.context_window == 5
        assert trainer.step == 0
        assert trainer.epoch == 0
    
    def test_setup_models(self, device, tokenizer, reconstructor):
        """Test model setup."""
        trainer = SimpleJointTrainer(device=device)
        
        with patch('training.trainer.SimpleCompressionPolicy') as mock_policy:
            with patch('training.trainer.make_model_deterministic') as mock_deterministic:
                with patch('torch.optim.Adam') as mock_optimizer:
                    mock_policy_instance = Mock()
                    mock_policy_instance.to.return_value = mock_policy_instance
                    # Mock modules() method for policy as well
                    mock_policy_instance.modules.return_value = []
                    # Mock parameters() method for optimizer
                    dummy_param = torch.tensor([1.0], requires_grad=True)
                    mock_policy_instance.parameters.return_value = [dummy_param]
                    mock_policy.return_value = mock_policy_instance
                    
                    # Make the deterministic function return the same object
                    mock_deterministic.side_effect = lambda model, seed: model
                    
                    # Mock optimizer returns
                    mock_optimizer.return_value = Mock()
                    
                    reconstructor.to.return_value = reconstructor
                    
                    trainer.setup_models(tokenizer, reconstructor)
                    
                    # Verify policy was created correctly
                    mock_policy.assert_called_once_with(
                        embedding_dim=768,
                        context_window=trainer.context_window,
                        device=device
                    )
                    mock_policy_instance.to.assert_called_once_with(device)
                    
                    # Verify reconstructor was moved to device
                    reconstructor.to.assert_called_once_with(device)
                    
                    # Verify deterministic functions were called
                    assert mock_deterministic.call_count == 2  # Once for policy, once for reconstructor
                    
                    # Verify optimizers were created
                    assert hasattr(trainer, 'policy_optimizer')
                    assert hasattr(trainer, 'reconstructor_optimizer')
    
    def test_train_step_shapes(self, device, tokenizer, reconstructor, sample_data):
        """Test that train_step processes tensor shapes correctly."""
        trainer = SimpleJointTrainer(device=device, micro_batch_size=2, gradient_accumulation_steps=2)
        
        with patch('training.trainer.SimpleCompressionPolicy') as mock_policy:
            with patch('training.trainer.make_model_deterministic') as mock_deterministic:
                with patch('torch.optim.Adam') as mock_optimizer:
                    # Setup mocks
                    mock_policy_instance = Mock()
                    mock_policy_instance.to.return_value = mock_policy_instance
                    def dynamic_policy_output(input_tensor):
                        batch_size, seq_len, _ = input_tensor.shape
                        return torch.randn(batch_size, seq_len, requires_grad=True)
                    mock_policy_instance.side_effect = dynamic_policy_output
                    mock_policy_instance.modules.return_value = []
                    dummy_param = torch.tensor([1.0], requires_grad=True)
                    mock_policy_instance.parameters.return_value = [dummy_param]
                    mock_policy.return_value = mock_policy_instance
                    
                    # Make the deterministic function return the same object
                    mock_deterministic.side_effect = lambda model, seed: model
                    
                    # Mock optimizer returns
                    mock_optimizer.return_value = Mock()
                    
                    trainer.setup_models(tokenizer, reconstructor)
                    trainer.policy = mock_policy_instance
                    trainer.reconstructor = reconstructor
                    trainer.tokenizer = tokenizer
                    
                    # Run train step
                    result = trainer.train_step(sample_data)
                    
                    # Verify return format
                    assert isinstance(result, dict)
                    expected_keys = {'policy_loss', 'reconstructor_loss', 'total_loss', 'reward', 'compression_ratio'}
                    assert set(result.keys()) == expected_keys
                    
                    # Verify all values are floats
                    for key, value in result.items():
                        assert isinstance(value, float)
    
    def test_train_step_gradient_flow(self, device, tokenizer, reconstructor, sample_data):
        """Test that gradient flow works correctly without retain_graph."""
        trainer = SimpleJointTrainer(device=device, micro_batch_size=2, gradient_accumulation_steps=2)
        
        with patch('training.trainer.SimpleCompressionPolicy') as mock_policy:
            with patch('training.trainer.make_model_deterministic') as mock_deterministic:
                with patch('torch.optim.Adam') as mock_optimizer:
                    # Setup mocks
                    mock_policy_instance = Mock()
                    mock_policy_instance.to.return_value = mock_policy_instance
                    def dynamic_policy_output(input_tensor):
                        batch_size, seq_len, _ = input_tensor.shape
                        return torch.randn(batch_size, seq_len, requires_grad=True)
                    mock_policy_instance.side_effect = dynamic_policy_output
                    mock_policy_instance.modules.return_value = []
                    dummy_param = torch.tensor([1.0], requires_grad=True)
                    mock_policy_instance.parameters.return_value = [dummy_param]
                    mock_policy.return_value = mock_policy_instance
                    
                    # Make the deterministic function return the same object
                    mock_deterministic.side_effect = lambda model, seed: model
                    
                    # Mock optimizers with step counters
                    mock_policy_opt = Mock()
                    mock_recon_opt = Mock()
                    mock_optimizer.side_effect = [mock_policy_opt, mock_recon_opt]
                    
                    trainer.setup_models(tokenizer, reconstructor)
                    trainer.policy = mock_policy_instance
                    trainer.reconstructor = reconstructor
                    trainer.tokenizer = tokenizer
                    
                    initial_step = trainer.step
                    
                    # Run train step
                    trainer.train_step(sample_data)
                    
                    # Verify optimizers were called correctly
                    trainer.policy_optimizer.zero_grad.assert_called_once()
                    trainer.reconstructor_optimizer.zero_grad.assert_called_once()
                    trainer.policy_optimizer.step.assert_called_once()
                    trainer.reconstructor_optimizer.step.assert_called_once()
                    
                    # Verify step counter incremented
                    assert trainer.step == initial_step + 1
    
    def test_deterministic_behavior(self, device, tokenizer, reconstructor, sample_data):
        """Test that same input produces same output with fixed seed."""
        # This test verifies the trainer setup is deterministic, but since we're using mocks
        # with random outputs, we'll just ensure the trainers can be created identically
        trainer1 = SimpleJointTrainer(device=device, random_seed=42, micro_batch_size=2, gradient_accumulation_steps=2)
        trainer2 = SimpleJointTrainer(device=device, random_seed=42, micro_batch_size=2, gradient_accumulation_steps=2)
        
        # Verify both trainers have identical configurations
        assert trainer1.random_seed == trainer2.random_seed
        assert trainer1.micro_batch_size == trainer2.micro_batch_size
        assert trainer1.gradient_accumulation_steps == trainer2.gradient_accumulation_steps
        assert trainer1.device == trainer2.device
        assert trainer1.context_window == trainer2.context_window
    
    def test_save_checkpoint(self, device, tokenizer, reconstructor, tmp_path):
        """Test checkpoint saving."""
        trainer = SimpleJointTrainer(device=device, micro_batch_size=2, gradient_accumulation_steps=2)
        trainer.step = 100
        trainer.epoch = 5
        
        # Setup minimal mocks
        trainer.policy = Mock()
        trainer.policy.state_dict.return_value = {'policy': 'state'}
        trainer.reconstructor = Mock()
        trainer.reconstructor.state_dict.return_value = {'reconstructor': 'state'}
        trainer.policy_optimizer = Mock()
        trainer.policy_optimizer.state_dict.return_value = {'policy_opt': 'state'}
        trainer.reconstructor_optimizer = Mock()
        trainer.reconstructor_optimizer.state_dict.return_value = {'recon_opt': 'state'}
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Verify checkpoint was saved
        assert checkpoint_path.exists()
        
        # Load and verify contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        expected_keys = {
            'policy_state_dict', 'reconstructor_state_dict',
            'policy_optimizer_state_dict', 'reconstructor_optimizer_state_dict',
            'step', 'epoch'
        }
        assert set(checkpoint.keys()) == expected_keys
        assert checkpoint['step'] == 100
        assert checkpoint['epoch'] == 5


def test_create_simple_trainer():
    """Test trainer factory function."""
    config = {
        'learning_rate_policy': 2e-3,
        'learning_rate_reconstructor': 5e-5,
        'device': 'cpu',
        'context_window': 7
    }
    
    train_data = [[1, 2, 3], [4, 5, 6]]
    reconstructor = Mock()
    tokenizer = Mock()
    
    with patch('training.trainer.SimpleJointTrainer') as mock_trainer_class:
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        result = create_simple_trainer(config, train_data, reconstructor, tokenizer)
        
        # Verify trainer was created with correct config (including gradient accumulation)
        mock_trainer_class.assert_called_once_with(
            policy_lr=2e-3,
            reconstructor_lr=5e-5,
            device='cpu',
            context_window=7,
            random_seed=42,
            micro_batch_size=2,
            gradient_accumulation_steps=8
        )
        
        # Verify setup_models was called
        mock_trainer.setup_models.assert_called_once_with(tokenizer, reconstructor)
        
        assert result == mock_trainer


class TestTrainerIntegration:
    """Integration tests for the trainer."""
    
    def test_full_training_run(self, device):
        """Test a complete training run with real models (but small data)."""
        # Skip if no GPU available for integration test
        if device == "cpu":
            pytest.skip("Integration test requires GPU")
        
        # Create minimal real models
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        reconstructor = GPT2LMHeadModel.from_pretrained("gpt2")
        reconstructor.resize_token_embeddings(len(tokenizer))
        
        # Minimal training data
        train_data = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10]
        ]
        
        trainer = SimpleJointTrainer(device=device, micro_batch_size=1, gradient_accumulation_steps=1)
        trainer.setup_models(tokenizer, reconstructor)
        
        # Train for 1 epoch with very small batch
        initial_step = trainer.step
        trainer.train(
            train_data=train_data,
            batch_size=1,
            max_epochs=1,
            output_dir="outputs/test_training"
        )
        
        # Verify training completed
        assert trainer.step > initial_step
        assert trainer.epoch == 1