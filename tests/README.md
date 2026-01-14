# Unit Tests for Gait Analysis System

This directory contains unit tests for the core components of the gait analysis system.

## Test Coverage

### Core Algorithm Tests

1. **test_v7_features.py** - V7 Pure 3D Feature Extraction
   - Tests all 10 biomechanical features
   - Validates feature ranges and types
   - Tests error handling for invalid inputs

2. **test_v8_classifier.py** - V8 ML-Enhanced Classifier
   - Tests logistic regression training
   - Validates prediction format
   - Tests feature importance extraction
   - Validates accuracy metrics

3. **test_mediapipe_processor.py** - MediaPipe CSV Processing
   - Tests 3D pose data loading
   - Validates coordinate format
   - Tests frame sequence continuity

## Running Tests

### Run all tests:
```bash
python -m pytest tests/
```

### Run specific test file:
```bash
python -m pytest tests/test_v7_features.py -v
```

### Run with coverage:
```bash
python -m pytest tests/ --cov=core --cov-report=html
```

### Run individual test:
```bash
python tests/test_v7_features.py
```

## Test Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov
```

## Test Data

Tests use synthetic data generated programmatically. For integration tests with real data, see the validation scripts in `archive/legacy_scripts/`.

## Expected Results

All tests should pass with the current implementation:
- **test_v7_features.py**: 7+ tests passing
- **test_v8_classifier.py**: 8+ tests passing
- **test_mediapipe_processor.py**: 5+ tests passing

## Adding New Tests

When adding new features:
1. Create a new test file: `test_<module_name>.py`
2. Follow the existing test structure
3. Use `setUp()` and `tearDown()` for test fixtures
4. Use descriptive test names: `test_<what_is_tested>`
5. Add assertions with clear messages

## Continuous Integration

Tests are designed to run in CI/CD pipelines. See `.github/workflows/` for automation setup.
