# Contributing to json-memory

Thank you for your interest in improving `json-memory`!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/dioncx/json-memory.git
   cd json-memory
   ```

2. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

We use `pytest` for testing.

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# Run with coverage (if pytest-cov is installed)
PYTHONPATH=. pytest --cov=json_memory tests/
```

## Code Quality

We use `black` for formatting and `mypy` for type checking.

```bash
# Format code
black json_memory/ tests/

# Type checking
mypy json_memory/
```

## Project Structure

- `json_memory/`: Core library code.
- `tests/`: Test suite.
- `benchmark.py`: Performance measurement scripts.
- `examples/`: Usage demonstrations.
