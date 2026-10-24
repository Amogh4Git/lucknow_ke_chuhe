# Environment Setup & Execution Guide

Follow these instructions to set up the Python development environment, install dependencies, run tests, and execute the application.

---

## 1. Verify Python Version

This project requires **Python 3.10** or newer.

Check your installed Python version:

```bash
python --version
```

Verify that the output displays `Python 3.10.x` or higher.

---

## 2. Create the Local Virtual Environment

Create an isolated virtual environment named `.venv` in the repository root directory:

```bash
python -m venv .venv
```

---

## 3. Activate the Virtual Environment

Activate the virtual environment according to your operating system:

### Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
```

*Note: If PowerShell blocks script execution, run:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Windows Command Prompt (CMD)
```cmd
.\.venv\Scripts\activate.bat
```

### Linux and macOS
```bash
source .venv/bin/activate
```

---

## 4. Install Dependencies

Upgrade `pip` and install all required production packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional: Install developer and static analysis tools:

```bash
python -m pip install -r requirements-dev.txt
```

---

## 5. Verify Package Installation

Confirm that all core packages import without errors:

```bash
python -c "import numpy, scipy, torch, sklearn, matplotlib, streamlit, pytest; print('All required packages imported successfully.')"
```

---

## 6. Run the Test Suite

Run all automated unit tests using pytest:

```bash
python -m pytest -v
```

To run a specific module test:

```bash
# Physics simulation tests
python -m pytest tests/test_physics.py -v

# Signal processing tests
python -m pytest tests/test_signal.py -v

# 1D-CNN model tests
python -m pytest tests/test_models.py -v

# Dashboard UI tests
python -m pytest tests/test_dashboard.py -v
```

---

## 7. Train and Evaluate the Model

Train the Multi-Task 1D-CNN model:

```bash
python src/models/train.py
```
*Weights save to `saved_models/clog_1dcnn.pth`.*

Evaluate model accuracy, F1-score, MAE, and noise degradation:

```bash
python src/models/evaluate.py
```

---

## 8. Run the Streamlit Dashboard

Start the real-time Digital Twin web application:

```bash
streamlit run src/dashboard/app.py
```

Open your browser to:
```text
http://localhost:8501
```

---

## 9. Troubleshooting Common Issues

### Issue 1: Script execution disabled on Windows PowerShell
**Symptom**: `.\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled.`
**Resolution**: Run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
Alternatively, invoke the virtual environment python directly:
```powershell
.\.venv\Scripts\python.exe -m pytest
```

### Issue 2: `ModuleNotFoundError: No module named 'src'`
**Symptom**: Python fails to locate `src` when running standalone scripts.
**Resolution**: Run commands from the repository root, or set `PYTHONPATH`:
```powershell
$env:PYTHONPATH = "."
```

### Issue 3: Port 8501 already in use by Streamlit
**Symptom**: Streamlit displays `Port 8501 is already in use`.
**Resolution**: Specify an alternative port:
```bash
streamlit run src/dashboard/app.py --server.port 8502
```
