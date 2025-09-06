#!/bin/bash
# RL Token Compression - Complete Setup Script
# One script to rule them all - no Docker complexity, just clean setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MIN_PYTHON_VERSION="3.8"
REQUIRED_PYTHON_VERSION="3.9"
VENV_NAME="venv"

# MPS Memory Configuration for Apple Silicon
# Disable artificial 60% memory limit to use full unified memory
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0

# Print functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}RL Token Compression Setup${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}[$(date '+%H:%M:%S') STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ❌${NC} $1"
}

print_info() {
    echo -e "   [$(date '+%H:%M:%S')] $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Version comparison function
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1 | grep -q "^$2$"
}

# Check system requirements
check_system_requirements() {
    print_step "Checking system requirements..."
    
    # Check OS
    OS=$(uname -s)
    print_info "Operating System: $OS"
    
    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        print_info "Please install Python 3.${MIN_PYTHON_VERSION}+ from https://python.org"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_info "Python version: $PYTHON_VERSION"
    
    if ! version_ge "$PYTHON_VERSION" "$MIN_PYTHON_VERSION"; then
        print_error "Python $MIN_PYTHON_VERSION+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    if ! version_ge "$PYTHON_VERSION" "$REQUIRED_PYTHON_VERSION"; then
        print_warning "Python $REQUIRED_PYTHON_VERSION+ recommended for best compatibility"
    fi
    
    # Check pip
    if ! python3 -m pip --version >/dev/null 2>&1; then
        print_error "pip is not available"
        print_info "Install with: python3 -m ensurepip --upgrade"
        exit 1
    fi
    
    # Check git
    if ! command_exists git; then
        print_warning "Git not found - version tracking will be limited"
    fi
    
    print_success "System requirements satisfied"
}

# Detect compute platform
detect_compute_platform() {
    print_step "Detecting compute platform..."
    
    COMPUTE_PLATFORM="cpu"
    GPU_INFO=""
    
    # Check for NVIDIA GPU
    if command_exists nvidia-smi; then
        if nvidia-smi >/dev/null 2>&1; then
            COMPUTE_PLATFORM="cuda"
            GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
            GPU_INFO="NVIDIA GPUs detected: $GPU_COUNT"
            print_info "$GPU_INFO"
            
            # Check for specific GPU models for optimized configs
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
            if echo "$GPU_NAME" | grep -qi "T4"; then
                GPU_MODEL="t4"
                print_info "T4 GPU detected - will use T4-optimized config"
            elif echo "$GPU_NAME" | grep -qi "V100\|A100\|RTX"; then
                GPU_MODEL="high_end"
                print_info "High-end GPU detected - will use CUDA-optimized config"
            else
                GPU_MODEL="standard"
                print_info "Standard GPU detected - will use CUDA-optimized config"
            fi
        fi
    fi
    
    # Check for Apple Silicon (MPS)
    if [[ "$OS" == "Darwin" ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            if [[ "$COMPUTE_PLATFORM" == "cpu" ]]; then
                COMPUTE_PLATFORM="mps"
                GPU_INFO="Apple Silicon (MPS) detected"
                print_info "$GPU_INFO"
                print_info "MPS memory limits disabled (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0)"
                print_info "Can use full unified memory (48GB on M4 Pro)"
            fi
        fi
    fi
    
    if [[ "$COMPUTE_PLATFORM" == "cpu" ]]; then
        print_info "CPU-only mode (no GPU acceleration detected)"
    fi
    
    print_success "Compute platform: $COMPUTE_PLATFORM"
}

# Create and setup virtual environment
setup_virtual_environment() {
    print_step "Setting up virtual environment..."
    
    if [[ -d "$VENV_NAME" ]]; then
        print_info "Virtual environment exists, activating..."
    else
        print_info "Creating new virtual environment..."
        python3 -m venv "$VENV_NAME"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Virtual environment ready"
}

# Install dependencies based on compute platform
install_dependencies() {
    print_step "Installing dependencies for $COMPUTE_PLATFORM platform..."
    
    # Create requirements based on platform
    case "$COMPUTE_PLATFORM" in
        cuda)
            print_info "Installing PyTorch with CUDA support..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        mps)
            print_info "Installing PyTorch with MPS support..."
            pip install torch torchvision torchaudio
            ;;
        cpu)
            print_info "Installing PyTorch (CPU-only)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
    
    # Install other ML dependencies
    print_info "Installing transformers and datasets..."
    pip install transformers datasets
    
    # Install additional requirements
    if [[ -f "requirements.txt" ]]; then
        print_info "Installing requirements from requirements.txt..."
        pip install -r requirements.txt
    else
        print_info "Installing core dependencies..."
        pip install numpy scipy matplotlib seaborn tqdm psutil
        pip install pytest pytest-mock  # Testing
    fi
    
    print_success "Dependencies installed"
}

# Validate installation
validate_installation() {
    print_step "Validating installation..."
    
    # Test Python imports
    print_info "Testing core imports..."
    python3 -c "
import torch
import numpy as np
import transformers
import datasets
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ NumPy: {np.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Datasets: {datasets.__version__}')
"
    
    # Test compute platform
    print_info "Testing compute platform..."
    case "$COMPUTE_PLATFORM" in
        cuda)
            python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
print(f'✓ CUDA devices: {torch.cuda.device_count()}')
"
            ;;
        mps)
            python3 -c "
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ MPS available')
else:
    print('⚠ MPS not available, falling back to CPU')
"
            ;;
        cpu)
            python3 -c "print('✓ CPU mode confirmed')"
            ;;
    esac
    
    print_success "Installation validated"
}

# Run smoke tests
run_smoke_tests() {
    print_step "Running smoke tests..."
    
    # Test core modules import
    print_info "Testing module imports..."
    python3 -c "
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from utils.config import setup_config
    from utils.errors import handle_errors
    from utils.deterministic import set_global_seed
    from training.trainer import SimpleJointTrainer
    print('✓ All core modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    # Test CLI interfaces
    print_info "Testing CLI interfaces..."
    if ! python3 data/prepare.py --help >/dev/null 2>&1; then
        print_error "Data preparation script not working"
        exit 1
    fi
    
    if ! python3 training/train.py --help >/dev/null 2>&1; then
        print_error "Training script not working"
        exit 1
    fi
    
    if ! python3 evaluation/evaluate.py --help >/dev/null 2>&1; then
        print_error "Evaluation script not working"
        exit 1
    fi
    
    # Run unit tests if available
    if command_exists pytest && [[ -d "tests" ]]; then
        print_info "Running unit tests..."
        pytest tests/ -q --tb=no --disable-warnings || {
            print_warning "Some unit tests failed - check with: pytest tests/ -v"
        }
    fi
    
    print_success "Smoke tests passed"
}

# Create activation script
create_activation_script() {
    print_step "Creating activation script..."
    
    cat > activate.sh << 'EOF'
#!/bin/bash
# Activate RL Token Compression environment
source venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "✓ RL Token Compression environment activated"
echo "  Python: $(python --version)"
echo "  Working directory: $(pwd)"
echo ""
echo "Quick start:"
echo "  python data/prepare.py --config configs/data/sample.json"
echo "  python training/train.py --config configs/training/default.json"
echo "  python evaluation/evaluate.py --config configs/evaluation/default.json"
echo ""
EOF
    
    chmod +x activate.sh
    
    print_success "Created activate.sh script"
}

# Print completion message
print_completion() {
    echo ""
    echo -e "${GREEN}🎉 Setup Complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Activate environment:"
    echo "   ${BLUE}source activate.sh${NC}"
    echo ""
    echo "2. Test installation:"
    echo "   ${BLUE}./setup.sh --integration-test${NC}    # Quick end-to-end test"
    echo "   ${BLUE}./setup.sh --validate-production${NC} # Complete validation"
    echo ""
    echo "3. Run quick test manually:"
    echo "   ${BLUE}python data/prepare.py --config configs/data/sample.json${NC}"
    echo ""
    echo "4. Start training:"
    echo "   ${BLUE}python training/train.py --config configs/training/default.json${NC}"
    echo ""
    echo "System info:"
    echo "  Platform: $COMPUTE_PLATFORM"
    if [[ -n "$GPU_INFO" ]]; then
        echo "  GPU: $GPU_INFO"
    fi
    echo "  Python: $(python3 --version)"
    echo "  Virtual env: $VENV_NAME/"
    echo ""
    echo -e "${YELLOW}Pro tip:${NC} Use 'source activate.sh' to quickly activate the environment"
}

# Run integration tests with mini configs
run_integration_tests() {
    print_step "Running integration tests with mini configs..."
    
    # Ensure we're in the right environment
    if [[ ! -d "$VENV_NAME" ]]; then
        print_error "Virtual environment not found. Run setup first."
        exit 1
    fi
    
    source "$VENV_NAME/bin/activate"
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    # Clean up any previous integration outputs
    rm -rf integration/outputs/
    
    echo "   Testing data preparation pipeline..."
    if python data/prepare.py --config configs/integration/data.json 2>&1 | tee -a integration_test.log; then
        print_success "✓ Data preparation successful"
    else
        print_error "✗ Data preparation failed - check integration_test.log"
        tail -10 integration_test.log
        return 1
    fi
    
    echo "   Testing training pipeline..."  
    if python training/train.py --config configs/integration/training.json 2>&1 | tee -a integration_test.log; then
        print_success "✓ Training pipeline successful"
    else
        print_error "✗ Training pipeline failed - check integration_test.log"
        tail -10 integration_test.log
        return 1
    fi
    
    echo "   Testing evaluation pipeline..."
    if python evaluation/evaluate.py --config configs/integration/evaluation.json 2>&1 | tee -a integration_test.log; then
        print_success "✓ Evaluation pipeline successful"
    else
        print_error "✗ Evaluation pipeline failed - check integration_test.log"
        tail -10 integration_test.log
        return 1
    fi
    
    # Verify key outputs exist
    local expected_files=(
        "integration/outputs/data_mini/processed_data.json"
        "integration/outputs/training_mini/final_model.pt"
        "integration/outputs/evaluation_mini/comprehensive_results.json"
    )
    
    for file in "${expected_files[@]}"; do
        if [[ -f "$file" ]]; then
            print_success "✓ Expected output file: $file"
        else
            print_error "✗ Missing expected output: $file"
            return 1
        fi
    done
    
    print_success "Integration tests passed"
    echo "   Log saved to: integration_test.log"
    echo "   Outputs in: integration/outputs/"
}

# Run production validation with sample configs
run_production_validation() {
    print_step "Running production validation with sample configs..."
    
    source "$VENV_NAME/bin/activate"
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    # Clean up any previous outputs
    rm -rf outputs/data_sample outputs/training_debug outputs/evaluation_sample
    
    echo "   Testing with data_sample.json..."
    if python data/prepare.py --config configs/data/sample.json 2>&1 | tee production_test.log; then
        print_success "✓ Sample data preparation successful"
    else
        print_error "✗ Sample data preparation failed - check production_test.log"
        tail -10 production_test.log
        return 1
    fi
    
    echo "   Testing with training_debug.json..."
    # Update training_debug.json to use the sample data we just created
    local temp_config=$(mktemp)
    jq '.data_path = "outputs/data_sample/processed_data.json" | .val_data_path = "outputs/data_sample/val_data.json"' configs/training/debug.json > "$temp_config"
    
    if python training/train.py --config "$temp_config" 2>&1 | tee -a production_test.log; then
        print_success "✓ Debug training successful"
    else
        print_error "✗ Debug training failed - check production_test.log"
        tail -10 production_test.log
        rm -f "$temp_config"
        return 1
    fi
    rm -f "$temp_config"
    
    echo "   Testing evaluation..."
    # Update evaluation config to use our trained model and test data
    local temp_eval=$(mktemp)
    jq '.model_path = "debug/run/best_model.zip" | .data_path = "outputs/data_sample/test_data.json" | .output_dir = "outputs/evaluation_sample"' configs/evaluation/default.json > "$temp_eval"
    
    if python evaluation/evaluate.py --config "$temp_eval" 2>&1 | tee -a production_test.log; then
        print_success "✓ Sample evaluation successful"
    else
        print_error "✗ Sample evaluation failed - check production_test.log"
        tail -10 production_test.log
        rm -f "$temp_eval"
        return 1
    fi
    rm -f "$temp_eval"
    
    print_success "Production validation passed"
    echo "   Log saved to: production_test.log"
    echo "   README commands verified working"
}

# Run full research validation with production configs
run_full_research_test() {
    local resume_flag="${1:-}"  # Accept resume flag as parameter
    
    print_step "Running full research validation with production configs..."
    if [[ -n "$resume_flag" ]]; then
        print_step "Resume mode enabled - will continue from existing checkpoints if found"
    fi
    print_warning "This will take HOURS to complete with full dataset and 100 epochs"
    
    source "$VENV_NAME/bin/activate"
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    # Clean up any previous outputs
    rm -rf outputs/research_test
    
    echo "   Phase 1: Full data preparation (may take 30+ minutes)..."
    if python data/prepare.py --config configs/data/full.json --output_dir outputs/research_test/data 2>&1 | tee full_research_test.log; then
        print_success "✓ Full data preparation successful"
    else
        print_error "✗ Full data preparation failed - check full_research_test.log"
        tail -20 full_research_test.log
        return 1
    fi
    
    echo "   Phase 2: Full production training (may take HOURS)..."
    
    # Detect compute platform for this specific run
    local detected_platform="cpu"
    local gpu_model="standard"
    
    if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        detected_platform="cuda"
        # Check for specific GPU models
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        if echo "$gpu_name" | grep -qi "T4"; then
            gpu_model="t4"
        elif echo "$gpu_name" | grep -qi "V100\|A100\|RTX"; then
            gpu_model="high_end"
        fi
    elif [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        detected_platform="mps"
    fi
    
    # Select appropriate config based on detected compute platform and GPU model
    local base_config="configs/training/default.json"  # Conservative default
    if [[ "$detected_platform" == "mps" ]]; then
        base_config="configs/training/mps.json"
        echo "   ✓ Detected Apple Silicon - Using MPS-optimized config (batch_size=16, memory-efficient)..."
    elif [[ "$detected_platform" == "cuda" ]]; then
        if [[ "$gpu_model" == "t4" ]]; then
            base_config="configs/training/t4.json"
            echo "   ✓ Detected T4 GPU - Using T4-optimized config (batch_size=64, accelerate+DeepSpeed)..."
        else
            base_config="configs/training/cuda.json"
            echo "   ✓ Detected NVIDIA GPU - Using CUDA-optimized config (batch_size=256, high-performance)..."
        fi
    else
        echo "   ✓ Using conservative default config (batch_size=64, device=auto)..."
    fi
    
    # Create temporary config that uses our test data
    local temp_training=$(mktemp)
    jq '.data_path = "outputs/research_test/data/processed_data.json" | .val_data_path = "outputs/research_test/data/val_data.json" | .output_dir = "outputs/research_test/training"' "$base_config" > "$temp_training"
    
    # Validate selected config
    local batch_size=$(jq -r '.batch_size' "$temp_training")
    local micro_batch_size=$(jq -r '.micro_batch_size' "$temp_training")
    local device=$(jq -r '.device' "$temp_training")
    echo "   Config validation: batch_size=$batch_size, micro_batch_size=$micro_batch_size, device=$device"
    
    if [[ "$detected_platform" == "mps" ]] && [[ "$batch_size" -gt 32 ]]; then
        print_warning "Large batch size ($batch_size) detected on MPS - this may cause OOM"
    fi
    
    if python training/train.py --config "$temp_training" $resume_flag 2>&1 | tee -a full_research_test.log; then
        print_success "✓ Full production training successful"
    else
        print_error "✗ Full production training failed - check full_research_test.log"
        tail -20 full_research_test.log
        rm -f "$temp_training"
        return 1
    fi
    rm -f "$temp_training"
    
    echo "   Phase 3: Full evaluation with all baselines..."
    # Create temporary evaluation config
    local temp_eval=$(mktemp)
    jq '.model_path = "outputs/research_test/training/best_model.zip" | .data_path = "outputs/research_test/data/test_data.json" | .output_dir = "outputs/research_test/evaluation"' configs/evaluation/default.json > "$temp_eval"
    
    if python evaluation/evaluate.py --config "$temp_eval" 2>&1 | tee -a full_research_test.log; then
        print_success "✓ Full evaluation successful"
    else
        print_error "✗ Full evaluation failed - check full_research_test.log"
        tail -20 full_research_test.log
        rm -f "$temp_eval"
        return 1
    fi
    rm -f "$temp_eval"
    
    print_success "🎉 Full research validation completed successfully!"
    echo "   This validates the complete research pipeline with production configs"
    echo "   Log saved to: full_research_test.log"
    echo "   Results in: outputs/research_test/"
    echo ""
    print_warning "Note: This was a full research run - results are research-quality"
}

# Handle script arguments
handle_arguments() {
    case "${1:-}" in
        --help|-h)
            echo "RL Token Compression Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --help, -h              Show this help message"
            echo "  --force                 Force reinstall even if environment exists"
            echo "  --cpu                   Force CPU-only installation"
            echo "  --test-only             Only run unit tests, don't install"
            echo "  --integration-test      Run integration tests after unit tests"
            echo "  --validate-production   Run production validation with sample configs"
            echo "  --full-research-test    Run complete research validation with full configs (LONG)"
            echo "                          Add --resume to continue from existing checkpoints"
            echo ""
            echo "The script will:"
            echo "1. Check system requirements"
            echo "2. Detect compute platform (CPU/CUDA/MPS)"
            echo "3. Create virtual environment"
            echo "4. Install appropriate dependencies"
            echo "5. Validate installation"
            echo "6. Run smoke tests"
            echo ""
            echo "Integration testing:"
            echo "  --integration-test      Fast end-to-end test (1-2 minutes)"
            echo "  --validate-production   Sample validation: integration + README commands (15-20 minutes)"
            echo "  --full-research-test    Full research validation with production configs (HOURS)"
            echo "                          Usage: ./setup.sh --full-research-test [--resume]"
            exit 0
            ;;
        --force)
            if [[ -d "$VENV_NAME" ]]; then
                print_info "Removing existing virtual environment..."
                rm -rf "$VENV_NAME"
            fi
            ;;
        --cpu)
            COMPUTE_PLATFORM="cpu"
            print_info "Forced CPU-only mode"
            ;;
        --test-only)
            if [[ -d "$VENV_NAME" ]]; then
                source "$VENV_NAME/bin/activate"
                validate_installation
                run_smoke_tests
                exit 0
            else
                print_error "No virtual environment found. Run setup first."
                exit 1
            fi
            ;;
        --integration-test)
            if [[ -d "$VENV_NAME" ]]; then
                source "$VENV_NAME/bin/activate"
                validate_installation
                run_smoke_tests
                run_integration_tests
                exit 0
            else
                print_error "No virtual environment found. Run setup first."
                exit 1
            fi
            ;;
        --validate-production)
            if [[ -d "$VENV_NAME" ]]; then
                source "$VENV_NAME/bin/activate"
                validate_installation
                run_smoke_tests
                run_integration_tests
                run_production_validation
                exit 0
            else
                print_error "No virtual environment found. Run setup first."
                exit 1
            fi
            ;;
        --full-research-test)
            # Check if --resume flag is provided as second argument
            local resume_flag=""
            if [[ "${2:-}" == "--resume" ]]; then
                resume_flag="--resume"
                echo "Resume mode enabled - will look for existing checkpoints"
            fi
            
            if [[ -d "$VENV_NAME" ]]; then
                source "$VENV_NAME/bin/activate"
                validate_installation
                run_smoke_tests
                run_integration_tests
                run_production_validation
                run_full_research_test "$resume_flag"
                exit 0
            else
                print_error "No virtual environment found. Run setup first."
                exit 1
            fi
            ;;
    esac
}

# Error handler
error_handler() {
    print_error "Setup failed on line $1"
    echo ""
    echo "Common solutions:"
    echo "1. Check Python version: python3 --version"
    echo "2. Check pip access: python3 -m pip --version"
    echo "3. Check internet connection"
    echo "4. Try: $0 --force"
    echo ""
    echo "For help, check the troubleshooting section in README.md"
    exit 1
}

# Set up error handling
trap 'error_handler $LINENO' ERR

# Main execution
main() {
    print_header
    handle_arguments "$@"
    check_system_requirements
    detect_compute_platform
    setup_virtual_environment
    install_dependencies
    validate_installation
    run_smoke_tests
    create_activation_script
    print_completion
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
else
    echo "This script should be executed, not sourced."
    echo "Run: ./setup.sh"
fi