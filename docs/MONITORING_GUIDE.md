# Real-Time Training Monitoring Guide

## Enhanced Logging Features

The system now includes comprehensive logging that tracks every step of the process. All scripts now output detailed progress information with timestamps.

## Quick Start Monitoring

### 1. Run with Live Output (Recommended)
```bash
# Full research test with live output (tee saves to log AND shows on screen)
./setup.sh --full-research-test

# This will show all progress in real-time while also saving to full_research_test.log
```

### 2. Monitor in Separate Terminal (If Running in Background)
```bash
# Terminal 1: Start training in background
./setup.sh --full-research-test > full_research_test.log 2>&1 &

# Terminal 2: Monitor live
tail -f full_research_test.log
```

## What You'll See

### Data Preparation Phase
```
[DATA-PREP] Starting dataset loading process: reddit
[DATA-PREP] Dataset loaded successfully in 45.23 seconds
[DATA-PREP] Dataset size: 1000000 examples
[DATA-PREP] Processed 1000 examples, 897 kept. Rate: 23.4/sec, ETA: 1847.3s
[DATA-PREP] Successfully processed 1000/50000 sequences
```

### Training Phase
```
[TRAINING] STARTING TRAINING LOOP
[TRAINING] Training will run 156 batches per epoch, 7800 total steps
[TRAINING] Processing batch 1/156 (sequences 0-32)
[TRAINING] Step 100/7800: Policy=0.5234, Recon=1.2341, Comp=0.421, Batch_time=2.31s, ETA=4.2h
[TRAINING] Epoch 1 complete in 467.23s - Avg Policy: 0.5234, Avg Recon: 1.2341, Avg Compression: 0.421
```

### Evaluation Phase
```
[EVALUATION] STARTING BASELINE EVALUATION
[EVALUATION] Test data loaded in 12.34 seconds (1000 sequences)
[EVALUATION] Starting evaluation of baseline 1/3: random
[EVALUATION] Baseline random completed in 23.45 seconds
[EVALUATION] Results for random: compression=0.325, quality=0.234
```

## Monitoring Commands

### Check if Process is Running
```bash
# Check if training is still running
ps aux | grep python | grep train

# Check if setup script is running  
ps aux | grep setup.sh
```

### Monitor System Resources
```bash
# Watch GPU usage (if CUDA)
watch -n 1 nvidia-smi

# Watch memory usage
watch -n 1 'free -h && df -h'

# Monitor CPU usage
htop
```

### Check Log File Growth
```bash
# Watch file size grow
watch -n 1 'ls -lh *.log'

# Count lines being added
watch -n 1 'wc -l *.log'

# Follow log file (alternative to method 2 above)
tail -f full_research_test.log
```

## Key Log Patterns to Watch For

### Normal Progress Signs
- ✅ `[DATA-PREP] Successfully processed X/Y sequences` 
- ✅ `[TRAINING] Step X/Y: Policy=... ETA=...`
- ✅ `[TRAINING] Epoch X complete in Ys`
- ✅ `[EVALUATION] Baseline X completed in Ys`

### Warning Signs
- ⚠️ `WARNING: Large batch size detected on MPS`
- ⚠️ `Tokenization failures: 100`
- ⚠️ `Memory: 90%+ usage`

### Error Signs
- ❌ `ERROR:` messages
- ❌ `Failed to evaluate`
- ❌ `CRASH DUMP SAVED`
- ❌ Process stops generating new lines

## Emergency Commands

### If Process Seems Stuck
```bash
# Check if it's really stuck (no new log lines in 5+ minutes)
tail -f full_research_test.log

# Check system resources
top
nvidia-smi  # for GPU

# Find the process
ps aux | grep python
```

### If You Need to Stop
```bash
# Find and kill the process gracefully
pkill -f "python.*train"
# or
kill -TERM <PID>

# Force kill if needed
kill -9 <PID>
```

## Files to Monitor

- `full_research_test.log` - Complete research test output
- `production_test.log` - Production validation output  
- `integration_test.log` - Integration test output

## Expected Timeline

For `--full-research-test`:
- **Data prep**: 30-60 minutes (depends on dataset size)
- **Training**: 2-8 hours (depends on hardware and epochs)
- **Evaluation**: 15-30 minutes (depends on baselines)

You should see constant progress updates throughout these phases.