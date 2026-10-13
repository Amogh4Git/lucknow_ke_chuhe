# Project Rules: Infrasonic Sewer Clog Detection Digital Twin

## Python Version Compatibility
- Maintain compatibility with Python 3.10+.
- Avoid using features or syntax incompatible with Python 3.10+.

## Code Quality and Architecture
- **Modular Code Design**: Structure code into cleanly decoupled modules under `src/` (`physics`, `signal`, `models`, `dashboard`).
- **Type Hints**: All function and class signatures must include explicit type hints (`typing` / built-in generics).
- **Documentation**: All public modules, functions, classes, and methods must have descriptive docstrings explaining their purpose, parameters, return types, and potential exceptions.

## Testing Standards
- Comprehensive unit tests are strictly required for:
  - Physics modules (`src/physics/`): Synthetic acoustic signal generation, pipe wave propagation, and attenuation.
  - Signal processing modules (`src/signal/`): Cross-correlation, time-difference-of-arrival (TDOA), spectral analysis, and feature extraction.
- All tests must be integrated with the `pytest` test suite in `tests/`.
- Ensure deterministic or seeded test cases for signal and acoustic simulation verification.
