"""
Unit tests for configuration system.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open
from types import SimpleNamespace

from utils.config import setup_config, save_config, resolve_device


class TestConfigSystem:
    """Test the configuration system."""
    
    def test_resolve_device_auto(self):
        """Test automatic device resolution."""
        # Test with no CUDA available
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = resolve_device("auto")
                assert device == "cpu"
        
        # Test with CUDA available
        with patch('torch.cuda.is_available', return_value=True):
            device = resolve_device("auto")
            assert device == "cuda"
        
        # Test with MPS available (Apple Silicon)
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = resolve_device("auto")
                assert device == "mps"
    
    def test_resolve_device_explicit(self):
        """Test explicit device specification."""
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("mps") == "mps"
    
    def test_save_config(self):
        """Test configuration saving."""
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "device": "cpu"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            try:
                save_config(config, f.name)
                
                # Read back and verify
                with open(f.name, 'r') as read_f:
                    saved_config = json.load(read_f)
                
                assert saved_config == config
                
            finally:
                os.unlink(f.name)
    
    def test_setup_config_defaults_only(self):
        """Test setup_config with only defaults."""
        default_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "device": "auto"
        }
        
        # Use SimpleNamespace instead of Mock to avoid internal attributes
        mock_args = SimpleNamespace(
            config=None,
            learning_rate=None,
            batch_size=None,
            device=None,
            resume=False,
            debug=False,
            max_epochs=None,
            output_dir=None,
            data_path=None
        )
        
        with patch('utils.config.argparse.ArgumentParser') as mock_parser:
            mock_parser.return_value.parse_args.return_value = mock_args
            
            config = setup_config(default_config, "Test Config")
            
            # Should keep defaults plus boolean flags that were set
            expected_config = default_config.copy()
            expected_config.update({
                'resume': False,
                'debug': False
            })
            assert config == expected_config
    
    def test_setup_config_with_file(self):
        """Test setup_config with config file."""
        default_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "device": "auto"
        }
        
        file_config = {
            "learning_rate": 0.002,
            "max_epochs": 100,
            "device": "cpu"
        }
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data=json.dumps(file_config))):
            with patch('os.path.exists', return_value=True):  # Mock file exists
                # Use SimpleNamespace for clean attribute access
                mock_args = SimpleNamespace(
                    config="config.json",
                    learning_rate=None,
                    batch_size=None,
                    device=None,
                    resume=False,
                    debug=False,
                    max_epochs=None,
                    output_dir=None,
                    data_path=None
                )
                
                with patch('utils.config.argparse.ArgumentParser') as mock_parser:
                    mock_parser.return_value.parse_args.return_value = mock_args
                    
                    config = setup_config(default_config, "Test Config")
                    
                    # Should merge default and file config plus boolean flags
                    expected_config = {
                        "learning_rate": 0.002,  # From file
                        "batch_size": 32,        # From default
                        "max_epochs": 100,       # From file
                        "device": "cpu",         # From file
                        'resume': False,          # Boolean flag
                        'debug': False            # Boolean flag
                    }
                    
                    assert config == expected_config
    
    def test_setup_config_with_cli_overrides(self):
        """Test setup_config with CLI overrides."""
        default_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "device": "auto"
        }
        
        # Use SimpleNamespace for CLI overrides
        mock_args = SimpleNamespace(
            config=None,
            learning_rate=0.005,
            batch_size=64,
            device="cuda",
            resume=False,
            debug=False,
            max_epochs=None,
            output_dir=None,
            data_path=None
        )
        
        with patch('utils.config.argparse.ArgumentParser') as mock_parser:
            mock_parser.return_value.parse_args.return_value = mock_args
            
            config = setup_config(default_config, "Test Config")
            
            # CLI args should override defaults
            expected_config = {
                "learning_rate": 0.005,
                "batch_size": 64,
                "device": "cuda",
                'resume': False,  # Boolean flags always included
                'debug': False
            }
            
            assert config == expected_config
    
    def test_setup_config_priority(self):
        """Test that CLI overrides take priority over file config."""
        default_config = {
            "learning_rate": 0.001,
            "batch_size": 32
        }
        
        file_config = {
            "learning_rate": 0.002,
            "batch_size": 16
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(file_config))):
            with patch('os.path.exists', return_value=True):  # Mock file exists
                # CLI overrides one parameter
                mock_args = SimpleNamespace(
                    config="config.json",
                    learning_rate=0.01,  # CLI override
                    batch_size=None,     # No override
                    device=None,
                    resume=False,
                    debug=False,
                    max_epochs=None,
                    output_dir=None,
                    data_path=None
                )
                
                with patch('utils.config.argparse.ArgumentParser') as mock_parser:
                    mock_parser.return_value.parse_args.return_value = mock_args
                    
                    config = setup_config(default_config, "Test Config")
                    
                    expected_config = {
                        "learning_rate": 0.01,   # CLI override
                        "batch_size": 16,        # From file
                        'resume': False,          # Boolean flag
                        'debug': False            # Boolean flag
                    }
                    
                    assert config == expected_config
    
    def test_config_validation(self):
        """Test that invalid JSON is handled gracefully."""
        default_config = {"batch_size": 32}
        
        # Mock invalid JSON that raises JSONDecodeError when opened
        mock_file = mock_open(read_data="invalid json")
        mock_file.return_value.read.side_effect = lambda: "invalid json"
        
        with patch('builtins.open', mock_file):
            with patch('os.path.exists', return_value=True):  # Mock file exists
                with patch('json.load', side_effect=json.JSONDecodeError("test", "invalid json", 0)):
                    mock_args = SimpleNamespace(
                        config="config.json",
                        batch_size=None,
                        learning_rate=None,
                        device=None,
                        resume=False,
                        debug=False,
                        max_epochs=None,
                        output_dir=None,
                        data_path=None
                    )
                    
                    with patch('utils.config.argparse.ArgumentParser') as mock_parser:
                        mock_parser.return_value.parse_args.return_value = mock_args
                        
                        # Should handle JSON decode error gracefully by falling back to defaults
                        config = setup_config(default_config, "Test Config")
                        assert config["batch_size"] == 32  # Should fall back to default
                        assert config["resume"] == False   # Boolean flags included
                        assert config["debug"] == False