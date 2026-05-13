# UniCircuitBench

This repository contains code for evaluating model performance on the **UniCircuitBench** benchmark. It includes scripts for testing different model types and a utility script to compute accuracy metrics for the generated results.

## Directory Structure
benchmark/
├── internvl-test.py # Code to test INTERV-L models
├── qwen3-vl-test.py # Code to test Qwen-3 VL models
├── test_generation.py # Code for circuit generation evaluation
├── test_understanding.py # Code for understanding tasks evaluation
accuracy.py # Script to compute accuracy for test outputs
README.md # This file

---

## Running Model Tests

Each test folder produces JSON files containing model predictions. These predictions are then evaluated using `accuracy.py`.

### Step 1: Run the tests

#### EXAMPLE:INTERNVL Test
```bash
cd benchmark/
python internvl-test.py

```
---

### Step 2: Run the tests
```bash
python accuracy.py
```
