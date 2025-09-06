"""
Proper error handling for RL Token Compression.

No crash dumps. No defensive programming. Just clean, actionable error messages.
"""

import sys
import traceback
from functools import wraps
from typing import Any, Callable


class TokenCompressionError(Exception):
    """Base exception for token compression errors."""

    pass


class ConfigurationError(TokenCompressionError):
    """Configuration-related errors."""

    pass


class DataError(TokenCompressionError):
    """Data loading/processing errors."""

    pass


class ModelError(TokenCompressionError):
    """Model loading/training errors."""

    pass


class DeviceError(TokenCompressionError):
    """Device-related errors (CUDA/MPS not available, etc.)."""

    pass


def handle_errors(error_context: str) -> Callable:
    """
    Clean error handling decorator.

    Converts common exceptions to actionable error messages.
    No crash dumps - just clear guidance for the user.

    Args:
        error_context: Context description (e.g., "data preparation", "training")
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)

            except ConfigurationError as e:
                print(f"\n❌ Configuration Error in {error_context}:")
                print(f"   {e}")
                print("   → Check your configuration file and CLI arguments")
                sys.exit(1)

            except DataError as e:
                print(f"\n❌ Data Error in {error_context}:")
                print(f"   {e}")
                print("   → Verify data files exist and are properly formatted")
                sys.exit(1)

            except ModelError as e:
                print(f"\n❌ Model Error in {error_context}:")
                print(f"   {e}")
                print("   → Check model paths and ensure models are compatible")
                sys.exit(1)

            except DeviceError as e:
                print(f"\n❌ Device Error in {error_context}:")
                print(f"   {e}")
                print("   → Try --device cpu or install proper GPU drivers")
                sys.exit(1)

            except FileNotFoundError as e:
                print(f"\n❌ File Not Found in {error_context}:")
                print(f"   {e}")
                print("   → Check file paths in your configuration")
                sys.exit(1)

            except PermissionError as e:
                print(f"\n❌ Permission Error in {error_context}:")
                print(f"   {e}")
                print("   → Check file permissions and output directory access")
                sys.exit(1)

            except ImportError as e:
                print(f"\n❌ Import Error in {error_context}:")
                print(f"   {e}")
                print("   → Install missing dependencies: pip install -r requirements.txt")
                sys.exit(1)

            except RuntimeError as e:
                error_msg = str(e).lower()
                if "cuda" in error_msg:
                    print(f"\n❌ CUDA Error in {error_context}:")
                    print(f"   {e}")
                    print("   → Try --device cpu or install CUDA drivers")
                    sys.exit(1)
                elif "mps" in error_msg:
                    print(f"\n❌ MPS Error in {error_context}:")
                    print(f"   {e}")
                    print("   → Try --device cpu or update to macOS 12.3+")
                    sys.exit(1)
                elif "out of memory" in error_msg or "memory" in error_msg:
                    print(f"\n❌ Memory Error in {error_context}:")
                    print(f"   {e}")
                    print("   → Reduce batch size or increase gradient accumulation")
                    sys.exit(1)
                else:
                    # Generic runtime error
                    print(f"\n❌ Runtime Error in {error_context}:")
                    print(f"   {e}")
                    print("   → Check your configuration and input data")
                    sys.exit(1)

            except KeyboardInterrupt:
                print(f"\n\n⚠️  {error_context.title()} interrupted by user")
                sys.exit(130)

            except Exception as e:
                print(f"\n❌ Unexpected Error in {error_context}:")
                print(f"   {e}")
                print(f"   Error type: {type(e).__name__}")
                print("\n   This is likely a bug. Please report it with:")
                print("   - Command you ran")
                print("   - Configuration used")
                print("   - Error message above")

                # Only show traceback in debug mode
                if "--debug" in sys.argv:
                    print("\n   Debug traceback:")
                    traceback.print_exc()

                sys.exit(1)

        return wrapper

    return decorator


def validate_file_exists(path: str, description: str) -> None:
    """Validate that a file exists with clear error message."""
    if not path:
        raise DataError(f"{description} path not provided")

    import os

    if not os.path.exists(path):
        raise DataError(f"{description} not found: {path}")

    if not os.path.isfile(path):
        raise DataError(f"{description} is not a file: {path}")


def validate_directory_writable(path: str, description: str) -> None:
    """Validate that directory is writable with clear error message."""
    import os

    # Create directory if it doesn't exist
    try:
        os.makedirs(path, exist_ok=True)
    except PermissionError:
        raise ConfigurationError(f"Cannot create {description} directory: {path}")

    # Test write access
    test_file = os.path.join(path, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except (PermissionError, OSError):
        raise ConfigurationError(f"{description} directory not writable: {path}")


def validate_device_available(device: str) -> str:
    """Validate device is available and return actual device string."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    elif device == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError("CUDA requested but not available")
        return "cuda"

    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceError("MPS requested but not available (requires macOS 12.3+ and Apple Silicon)")
        return "mps"

    elif device == "cpu":
        return "cpu"

    else:
        raise ConfigurationError(f"Unknown device: {device}. Use 'auto', 'cpu', 'cuda', or 'mps'")


def check_memory_requirements(device: str, batch_size: int, sequence_length: int) -> None:
    """Check if system has enough memory for given parameters."""
    import torch

    # Rough memory estimation (in GB)
    # GPT-2 base model: ~500MB
    # Policy network: ~10MB
    # Batch processing: batch_size * sequence_length * 4 bytes * safety_factor

    model_memory = 0.5  # GB
    batch_memory = batch_size * sequence_length * 4 * 2 / (1024**3)  # GB, 2x safety factor
    total_memory = model_memory + batch_memory

    if device == "cuda":
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if total_memory > gpu_memory * 0.8:  # 80% safety margin
                raise DeviceError(
                    f"Insufficient GPU memory. Need ~{total_memory:.1f}GB, have {gpu_memory:.1f}GB. "
                    f"Reduce batch_size or sequence length."
                )
        except Exception:
            # Can't check GPU memory, proceed with warning
            pass

    elif device == "mps":
        # MPS shares system memory, rougher estimate
        import psutil

        system_memory = psutil.virtual_memory().total / (1024**3)  # GB
        if total_memory > system_memory * 0.6:  # 60% safety margin
            raise DeviceError(
                f"Insufficient system memory for MPS. Need ~{total_memory:.1f}GB, have {system_memory:.1f}GB. "
                f"Reduce batch_size or use gradient accumulation."
            )

    # For CPU, assume it will work but warn if very large
    if total_memory > 8.0:  # 8GB threshold
        print(f"⚠️  Large memory requirement detected (~{total_memory:.1f}GB). Training may be slow.")


def suggest_batch_size(device: str, target_batch_size: int) -> dict:
    """Suggest optimal batch size configuration based on device."""
    suggestions = {}

    if device == "cpu":
        # CPU can handle larger batches but slower
        suggestions = {
            "batch_size": min(target_batch_size, 32),
            "gradient_accumulation_steps": max(1, target_batch_size // 32),
            "reason": "CPU training works better with moderate batch sizes",
        }

    elif device == "mps":
        # MPS benefits from smaller micro batches
        micro_batch = min(8, target_batch_size)
        grad_accum = target_batch_size // micro_batch
        suggestions = {
            "batch_size": target_batch_size,
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": grad_accum,
            "reason": "MPS works better with small micro batches and gradient accumulation",
        }

    elif device == "cuda":
        # CUDA can typically handle larger batches
        suggestions = {
            "batch_size": target_batch_size,
            "reason": "CUDA should handle the requested batch size",
        }

    return suggestions
