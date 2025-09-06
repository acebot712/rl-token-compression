"""
Unit tests for model components.
"""

import pytest
import torch

from models.agent import SimpleCompressionPolicy


class TestSimpleCompressionPolicy:
    """Test the SimpleCompressionPolicy model."""

    def test_init(self):
        """Test policy initialization."""
        policy = SimpleCompressionPolicy(embedding_dim=768, context_window=5, hidden_dim=256, device="cpu")

        assert policy.embedding_dim == 768
        assert policy.context_window == 5
        assert policy.hidden_dim == 256
        assert policy.device == "cpu"

    def test_forward_shape(self):
        """Test forward pass produces correct output shapes."""
        batch_size, seq_len, embed_dim = 2, 10, 768

        policy = SimpleCompressionPolicy(embedding_dim=embed_dim, context_window=5, device="cpu")

        # Create input tensor
        embeddings = torch.randn(batch_size, seq_len, embed_dim)

        # Forward pass
        output = policy(embeddings)

        # Check output shape
        expected_shape = (batch_size, seq_len)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_forward_values_range(self):
        """Test that forward pass produces reasonable output values."""
        batch_size, seq_len, embed_dim = 2, 8, 768

        policy = SimpleCompressionPolicy(embedding_dim=embed_dim, device="cpu")
        embeddings = torch.randn(batch_size, seq_len, embed_dim)

        output = policy(embeddings)

        # After sigmoid, values should be in [0, 1]
        probs = torch.sigmoid(output)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

        # Output should be reasonable (not all zeros or ones)
        assert not torch.all(probs < 0.01)  # Not all near zero
        assert not torch.all(probs > 0.99)  # Not all near one

    def test_context_window_effect(self):
        """Test that different context windows affect output."""
        batch_size, seq_len, embed_dim = 1, 10, 768
        embeddings = torch.randn(batch_size, seq_len, embed_dim)

        # Same random seed for fair comparison
        torch.manual_seed(42)
        policy1 = SimpleCompressionPolicy(embedding_dim=embed_dim, context_window=3, device="cpu")
        output1 = policy1(embeddings)

        torch.manual_seed(42)
        policy2 = SimpleCompressionPolicy(embedding_dim=embed_dim, context_window=7, device="cpu")
        output2 = policy2(embeddings)

        # Different context windows should produce different outputs
        assert not torch.allclose(output1, output2, atol=1e-4)

    def test_deterministic_behavior(self):
        """Test that same input produces same output."""
        batch_size, seq_len, embed_dim = 2, 6, 768
        embeddings = torch.randn(batch_size, seq_len, embed_dim)

        # Create two identical policies
        torch.manual_seed(42)
        policy1 = SimpleCompressionPolicy(embedding_dim=embed_dim, device="cpu")

        torch.manual_seed(42)
        policy2 = SimpleCompressionPolicy(embedding_dim=embed_dim, device="cpu")

        # Both should produce same output
        output1 = policy1(embeddings)
        output2 = policy2(embeddings)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        batch_size, seq_len, embed_dim = 2, 5, 768

        policy = SimpleCompressionPolicy(embedding_dim=embed_dim, device="cpu")
        embeddings = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

        output = policy(embeddings)
        loss = output.sum()  # Simple loss for gradient test

        # Backward pass
        loss.backward()

        # Check that policy parameters have gradients
        for param in policy.parameters():
            assert param.grad is not None, "Parameter has no gradient"
            assert not torch.all(param.grad == 0), "All gradients are zero"

        # Check that input gradients exist
        assert embeddings.grad is not None, "Input embeddings have no gradient"

    def test_device_handling(self):
        """Test that model moves to correct device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        policy = SimpleCompressionPolicy(embedding_dim=768, device="cuda")
        policy = policy.to("cuda")

        # Check that parameters are on the right device
        for param in policy.parameters():
            assert param.device.type == "cuda", f"Parameter on wrong device: {param.device}"

        # Test forward pass with CUDA tensors
        embeddings = torch.randn(1, 5, 768, device="cuda")
        output = policy(embeddings)

        assert output.device.type == "cuda", f"Output on wrong device: {output.device}"

    def test_eval_vs_train_mode(self):
        """Test behavior difference between training and eval modes."""
        batch_size, seq_len, embed_dim = 1, 8, 768
        embeddings = torch.randn(batch_size, seq_len, embed_dim)

        policy = SimpleCompressionPolicy(embedding_dim=embed_dim, device="cpu")

        # Training mode
        policy.train()
        train_output = policy(embeddings)

        # Eval mode
        policy.eval()
        eval_output = policy(embeddings)

        # Outputs should be identical (no dropout in this simple model)
        assert torch.allclose(train_output, eval_output, atol=1e-6)

    def test_parameter_count(self):
        """Test that model has expected number of parameters."""
        policy = SimpleCompressionPolicy(embedding_dim=768, context_window=5, hidden_dim=256, device="cpu")

        total_params = sum(p.numel() for p in policy.parameters())

        # Should be roughly 1M parameters (check within reasonable range)
        assert 500_000 < total_params < 2_000_000, f"Unexpected parameter count: {total_params}"

    def test_batch_size_independence(self):
        """Test that model works with different batch sizes."""
        embed_dim = 768
        seq_len = 6

        policy = SimpleCompressionPolicy(embedding_dim=embed_dim, device="cpu")

        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            embeddings = torch.randn(batch_size, seq_len, embed_dim)
            output = policy(embeddings)

            expected_shape = (batch_size, seq_len)
            assert output.shape == expected_shape, f"Failed for batch_size={batch_size}"

    def test_sequence_length_independence(self):
        """Test that model works with different sequence lengths."""
        embed_dim = 768
        batch_size = 2

        policy = SimpleCompressionPolicy(embedding_dim=embed_dim, device="cpu")

        # Test with different sequence lengths
        for seq_len in [1, 5, 10, 20]:
            embeddings = torch.randn(batch_size, seq_len, embed_dim)
            output = policy(embeddings)

            expected_shape = (batch_size, seq_len)
            assert output.shape == expected_shape, f"Failed for seq_len={seq_len}"
