# RL Token Compression - Professional Makefile
# Simple, reliable, gets the job done. No BS.
# Commands auto-setup environment - just run what you want.

SHELL := /bin/bash
.DEFAULT_GOAL := help
.ONESHELL:

# Configuration - override these as needed
VENV_NAME ?= venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip
PYTEST := $(VENV_NAME)/bin/pytest
RUFF := $(VENV_NAME)/bin/ruff

# Platform detection - let's not overcomplicate this
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Default compute platform
PLATFORM := cpu
ifeq ($(shell command -v nvidia-smi >/dev/null 2>&1 && echo cuda),cuda)
	PLATFORM := cuda
else ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
	PLATFORM := mps
endif

# Output directories
DATA_DIR := outputs/data
TRAIN_DIR := outputs/training
EVAL_DIR := outputs/evaluation
PLOTS_DIR := outputs/plots

# Color output because we're not animals
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

define print_info
	@echo -e "$(BLUE)[INFO]$(NC) $(1)"
endef

define print_success
	@echo -e "$(GREEN)[SUCCESS]$(NC) $(1)"
endef

define print_warning
	@echo -e "$(YELLOW)[WARNING]$(NC) $(1)"
endef

define print_error
	@echo -e "$(RED)[ERROR]$(NC) $(1)"
endef

# Phony targets
.PHONY: help setup clean test lint format install-deps
.PHONY: data-sample data-full train-debug train-full train-sample evaluate
.PHONY: integration-test production-test pipeline-full-resume
.PHONY: ensure-venv ensure-sample-data ensure-full-data ensure-trained-model ensure-evaluation all

##@ Help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Main Commands
# All commands auto-setup the environment - just run what you want

##@ Development
clean: ## Clean build artifacts and temporary files
	$(call print_info,"Cleaning build artifacts...")
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -f *.log
	$(call print_success,"Clean complete")

clean-all: clean ## Clean everything including outputs and models
	$(call print_warning,"Removing all outputs and models...")
	@rm -rf outputs/ debug/ integration/
	@rm -rf models/reconstructor/fine-tuned/
	@rm -rf models/agent/output/
	$(call print_warning,"All outputs cleaned")

test: ensure-venv ## Run component unit tests with detailed output and tracebacks
	$(call print_info,"Running unit tests...")
	@$(PYTEST) tests/ -v --tb=short
	$(call print_success,"Unit tests passed")


lint: ensure-venv ## Run linting (if ruff is available)
	$(call print_info,"Running linter...")
	@if [ -f "$(RUFF)" ]; then \
		$(RUFF) check . || echo -e "$(YELLOW)[WARNING]$(NC) Linting issues found"; \
	else \
		echo -e "$(YELLOW)[WARNING]$(NC) Ruff not installed, skipping lint"; \
	fi

format: ensure-venv ## Format code (if ruff is available)
	$(call print_info,"Formatting code...")
	@if [ -f "$(RUFF)" ]; then \
		$(RUFF) format . && echo -e "$(GREEN)[SUCCESS]$(NC) Code formatted"; \
	else \
		echo -e "$(YELLOW)[WARNING]$(NC) Ruff not installed, skipping format"; \
	fi

##@ Data Preparation
data-sample: ensure-venv ## Prepare sample dataset
	$(call print_info,"Preparing sample data...")
	@mkdir -p $(DATA_DIR)
	@$(PYTHON) data/prepare.py --config configs/data/sample.json
	$(call print_success,"Sample data prepared")

data-full: ensure-venv ## Prepare full dataset (takes time!)
	$(call print_info,"Preparing full dataset (this will take a while)...")
	@mkdir -p $(DATA_DIR)
	@$(PYTHON) data/prepare.py --config configs/data/full.json
	$(call print_success,"Full dataset prepared")

##@ Training
train-debug: ensure-venv ensure-sample-data ## Quick debug training run
	$(call print_info,"Starting debug training...")
	@mkdir -p $(TRAIN_DIR)
	@$(PYTHON) training/train.py --config configs/training/debug.json
	$(call print_success,"Debug training completed")

train-sample: data-sample ## Train on sample data
	$(call print_info,"Training on sample data...")
	@mkdir -p $(TRAIN_DIR)
	@$(PYTHON) training/train.py --config configs/training/debug.json
	$(call print_success,"Sample training completed")

train-full: ensure-venv ensure-full-data ## Full production training (long!)
	$(call print_info,"Starting full production training...")
	$(call print_warning,"This will take HOURS - consider using screen/tmux")
	@mkdir -p $(TRAIN_DIR)
	@if [ "$(PLATFORM)" = "mps" ]; then \
		$(call print_info,"Using MPS-optimized config for Apple Silicon"); \
		$(PYTHON) training/train.py --config configs/training/mps.json; \
	elif [ "$(PLATFORM)" = "cuda" ]; then \
		$(call print_info,"Using CUDA-optimized config"); \
		$(PYTHON) training/train.py --config configs/training/cuda.json; \
	else \
		$(call print_info,"Using default config"); \
		$(PYTHON) training/train.py --config configs/training/default.json; \
	fi
	$(call print_success,"Production training completed")

train-resume: ensure-venv ## Resume training from checkpoint
	$(call print_info,"Resuming training from checkpoint...")
	@$(PYTHON) training/train.py --resume
	$(call print_success,"Training resumed")

##@ Evaluation
evaluate: ensure-venv ensure-trained-model ## Evaluate trained model
	$(call print_info,"Running evaluation...")
	@mkdir -p $(EVAL_DIR)
	@$(PYTHON) evaluation/evaluate.py --config configs/evaluation/default.json
	$(call print_success,"Evaluation completed")

plots: ensure-evaluation ## Generate visualization plots
	$(call print_info,"Generating plots...")
	@mkdir -p $(PLOTS_DIR)
	@$(PYTHON) plots/visualize.py --results_path $(EVAL_DIR)/evaluation_results.json --output_dir $(PLOTS_DIR)
	$(call print_success,"Plots generated")

##@ Testing and Validation
integration-test: ensure-venv ## Run fast end-to-end pipeline tests with minimal data
	$(call print_info,"Running integration tests...")
	@./setup.sh --integration-test
	$(call print_success,"Integration tests passed")

production-test: ensure-venv ## Validate production configs and environment setup
	$(call print_info,"Running production validation...")
	@./setup.sh --validate-production
	$(call print_success,"Production validation passed")

##@ Pipeline Workflows
# Three clear pipeline options: sample (fast), debug (minimal), full (production)
# Use pipeline-full for complete research runs - no confusion, no duplication
pipeline-sample: ## Complete sample pipeline (data -> train -> evaluate)
	$(MAKE) data-sample
	$(MAKE) train-sample  
	$(MAKE) evaluate
	$(call print_success,"Sample pipeline completed")

pipeline-debug: ## Debug pipeline with minimal data
	$(MAKE) train-debug
	$(MAKE) evaluate
	$(call print_success,"Debug pipeline completed")

pipeline-full: ## Full production pipeline (VERY LONG!)
	$(call print_warning,"This is the full production pipeline - will take HOURS")
	@read -p "Are you sure you want to run the full pipeline? (y/N): " confirm && [ "$$confirm" = "y" ]
	$(MAKE) data-full
	$(MAKE) train-full
	$(MAKE) evaluate
	$(MAKE) plots
	$(call print_success,"Full pipeline completed")

pipeline-full-resume: ## Resume full pipeline from checkpoint
	$(call print_info,"Resuming full pipeline from checkpoint...")
	$(MAKE) train-resume
	$(MAKE) evaluate
	$(MAKE) plots
	$(call print_success,"Full pipeline resumed")

# Internal dependency management - auto-setup happens transparently
# These targets ensure prerequisites exist by creating them if missing.
# Users don't need to think about these - they just work.
ensure-venv:
	@if [ ! -d "$(VENV_NAME)" ] || [ ! -f "$(VENV_NAME)/pyvenv.cfg" ]; then \
		echo -e "$(BLUE)[INFO]$(NC) Virtual environment missing, setting up..."; \
		./setup.sh; \
	elif ! $(PYTHON) -c "import torch, transformers, stable_baselines3" >/dev/null 2>&1; then \
		echo -e "$(BLUE)[INFO]$(NC) Dependencies missing or incomplete, setting up..."; \
		./setup.sh; \
	fi

ensure-sample-data: ensure-venv
	@if [ ! -f "outputs/data_sample/processed_data.json" ] && [ ! -f "outputs/research_test/data/processed_data.json" ]; then \
		echo -e "$(BLUE)[INFO]$(NC) No sample training data found, preparing..."; \
		$(MAKE) data-sample; \
	fi

ensure-full-data: ensure-venv
	@if [ ! -f "outputs/data_full/processed_data.json" ] && [ ! -f "outputs/research_test/data/processed_data.json" ]; then \
		echo -e "$(BLUE)[INFO]$(NC) No full training data found, preparing..."; \
		$(MAKE) data-full; \
	fi

ensure-trained-model: ensure-venv ensure-sample-data
	@if [ ! -f "debug/run/best_model.zip" ] && [ ! -f "outputs/training/best_model.zip" ]; then \
		echo -e "$(BLUE)[INFO]$(NC) No trained model found, running sample training..."; \
		$(MAKE) train-sample; \
	fi

ensure-evaluation: ensure-trained-model
	@if [ ! -f "$(EVAL_DIR)/evaluation_results.json" ]; then \
		echo -e "$(BLUE)[INFO]$(NC) No evaluation results found, running evaluation..."; \
		$(MAKE) evaluate; \
	fi

##@ Status and Information
status: ## Show environment and project status
	$(call print_info,"Project Status")
	@echo "Platform: $(PLATFORM)"
	@echo "Virtual env: $(if $(wildcard $(VENV_NAME)),✓ Present,✗ Missing)"
	@echo "Training data: $(if $(wildcard outputs/data*/processed_data.json),✓ Present,✗ Missing)"
	@echo "Trained model: $(if $(wildcard debug/run/best_model.zip outputs/training/best_model.zip),✓ Present,✗ Missing)"
	@echo "Config files: $(shell find configs -name '*.json' | wc -l | tr -d ' ') files"
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "Python: $$($(PYTHON) --version)"; \
		echo "PyTorch: $$($(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"; \
	fi

info: status ## Show detailed project information
	$(call print_info,"Detailed Project Information")
	@echo ""
	@echo "Directory Structure:"
	@ls -la | grep ^d | awk '{print "  " $$9}'
	@echo ""
	@echo "Available Configs:"
	@find configs -name '*.json' | sort | sed 's|^|  |'
	@echo ""
	@echo "Recent Activity:"
	@if [ -f "integration_test.log" ]; then echo "  Last integration test: $$(stat -f '%Sm' integration_test.log)"; fi
	@if [ -f "production_test.log" ]; then echo "  Last production test: $$(stat -f '%Sm' production_test.log)"; fi

##@ All-in-one Targets
all: pipeline-sample ## Default: complete sample pipeline

dev: data-sample train-debug evaluate ## Developer workflow: sample data -> debug train -> evaluate

ci: test integration-test ## CI workflow: unit tests -> integration tests

##@ Advanced Setup (for power users)
# These commands are available but hidden from normal usage
# Most users should never need these - environment auto-sets up
setup: ## Complete environment setup (runs setup.sh)
	$(call print_info,"Setting up environment...")
	@./setup.sh
	$(call print_success,"Environment ready")

setup-force: ## Force reinstall environment
	$(call print_info,"Force reinstalling environment...")
	@./setup.sh --force
	$(call print_success,"Environment reinstalled")

setup-cpu: ## Setup with CPU-only PyTorch
	$(call print_info,"Setting up CPU-only environment...")
	@./setup.sh --cpu
	$(call print_success,"CPU-only environment ready")

install-deps: ensure-venv ## Install/update dependencies only
	$(call print_info,"Installing dependencies...")
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	$(call print_success,"Dependencies installed")

# Environment variables for consistent behavior
export PYTHONPATH := $(shell pwd):$(PYTHONPATH)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO := 0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO := 0.0