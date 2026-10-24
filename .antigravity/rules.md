# Antigravity Project Rules: Infrasonic Sewer Clog Detection Digital Twin

## 1. Python Compatibility
- All project code must support Python 3.10 or newer.
- Use standard Python packaging and import conventions.
- Avoid deprecated APIs and platform-specific assumptions.

## 2. Code Quality
- Follow PEP 8 style conventions across all modules.
- Use clear type hints for functions, methods, and variables.
- Add concise, meaningful docstrings to modules, classes, and public functions.
- Prefer small, reusable, single-responsibility modules.
- Keep physics simulation, signal processing, machine learning, and dashboard logic separate.
- Use meaningful constants instead of unexplained magic numbers.
- Handle invalid inputs and numerical edge cases explicitly.

## 3. Scientific and Signal-Processing Practices
- Document signal units, sampling frequency, assumptions, and parameters.
- Use deterministic random seeds in tests.
- Keep synthetic data generation reproducible.
- Clearly distinguish simulated data from real sensor measurements.
- Do not present synthetic model outputs as validated real-world clog detections.

## 4. Testing Requirements
- Unit tests are mandatory for `src/physics/` and `src/signal/`.
- Use pytest as the testing framework.
- Test normal inputs, invalid inputs, boundary conditions, and numerical behavior.
- Keep tests independent of external services and physical hardware.
- Ensure tests are deterministic and suitable for repeated local execution.
- Run the full test suite before you report task completion.

## 5. Documentation Requirements
- Maintain `README.md` with project purpose, architecture overview, setup steps, and test commands.
- Document the responsibilities of each module.
- Update documentation when the repository structure or setup changes.
- Never document a command as successful unless it has executed and succeeded.

## 6. Development Boundaries
- Do not introduce unnecessary dependencies.
- Inspect existing files before you modify them.
- Keep generated datasets, model checkpoints, virtual environments, caches, and secrets out of version control.
- Request user confirmation before you make destructive changes.
