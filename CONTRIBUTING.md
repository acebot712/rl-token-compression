# Contributing to RL Token Compression

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/rl-token-compression.git`
3. Set up the environment:
   ```bash
   ./setup.sh
   source activate.sh
   ```

## Development Workflow

### Running Tests

Test individual components:
```bash
python tests/test_data_prepare.py
python tests/test_models.py
python tests/test_trainer.py
python tests/test_evaluation.py
```

### Code Style

This project values clarity over cleverness:
- Write simple, readable code
- Add comments explaining WHY, not WHAT
- Keep functions focused and short
- Use type hints for function signatures
- Follow PEP 8 (run `black .` for formatting)

### Adding New Baselines

1. Inherit from `CompressionBaseline` in `evaluation/baselines.py`
2. Implement the `compress()` method
3. Add tests demonstrating your baseline works
4. Update evaluation configs to include it
5. Show it beats or is competitive with existing baselines

### Adding New Metrics

1. Add metric computation in `evaluation/evaluator.py`
2. Ensure metric is fast to compute (cache if needed)
3. Add statistical significance testing
4. Update visualization code to display it

### Training Improvements

1. Test on small data first (`configs/integration/training.json`)
2. Profile memory usage and training speed
3. Document any new hyperparameters
4. Ensure backward compatibility with existing configs

## Pull Request Guidelines

**Before submitting:**
1. Run all tests and ensure they pass
2. Add tests for new functionality
3. Update documentation (README, CLAUDE.md)
4. Ensure code follows project style
5. Include performance comparisons for algorithmic changes

**PR Description should include:**
- What problem does this solve?
- What are the measurable improvements?
- How did you test it?
- Any breaking changes?

## Performance Standards

- Baselines should complete 1000 sequences in < 10 seconds
- Training should support batch sizes down to 8 for limited memory
- Evaluation should be reproducible with fixed seeds
- Memory usage should be documented for major changes

## What We're Looking For

**High Priority:**
- Faster baseline implementations
- Better reconstruction quality metrics
- Memory-efficient training techniques
- Improved RL reward functions
- Comprehensive benchmarking

**Nice to Have:**
- Support for additional LLM architectures
- Distributed training improvements
- Better visualization tools
- Documentation improvements

**Not Interested In:**
- Premature optimizations without benchmarks
- Complex architectures without clear wins
- Adding dependencies without strong justification
- Breaking changes without migration path

## Questions?

Open an issue with the label `question`. Be specific about what you're trying to do and what you've already tried.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
