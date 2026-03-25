# Mac Setup Guide (Visual Studio Code)

Complete guide to run the Tree Crown Pipeline on macOS using Visual Studio Code.

## Prerequisites

- macOS 10.15 or later
- Homebrew installed
- Visual Studio Code installed

## Step-by-Step Installation

### 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3.9+

```bash
brew install python@3.9
```

Verify installation:
```bash
python3 --version  # Should show 3.9.x or higher
```

### 3. Install Git

```bash
brew install git
```

### 4. Clone the Repository

```bash
cd ~/Desktop  # or wherever you want
git clone https://github.com/yourusername/tree-crown-pipeline.git
cd tree-crown-pipeline
```

### 5. Set Up Python Environment

#### Option A: Using venv (Simple)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Conda (Recommended for ML projects)

```bash
# Install Miniconda
brew install --cask miniconda

# Initialize conda (restart terminal after this)
conda init zsh  # or bash if you use bash

# Create environment
conda create -n tree-crown python=3.9
conda activate tree-crown

# Install PyTorch (CPU version for Mac)
conda install pytorch torchvision -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### 6. Configure VS Code

#### Install VS Code Extensions

1. Open VS Code
2. Go to Extensions (Cmd+Shift+X)
3. Install:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft)
   - **Pylance** (by Microsoft)

#### Set Python Interpreter

1. Open the project in VS Code:
   ```bash
   code .
   ```

2. Press `Cmd+Shift+P`
3. Type: "Python: Select Interpreter"
4. Choose your environment:
   - For venv: `./venv/bin/python`
   - For conda: `tree-crown (conda)`

### 7. Configure Your Data Paths

Edit `tree_crown_pipeline.py`:

```python
class Config:
    # Update these paths to your data
    ORTHO_FOLDER = '/Users/yourname/data/orthomosaics'
    POLY_FOLDER = '/Users/yourname/data/polygons'
    STEP1_OUTPUT = '/Users/yourname/data/step1_output'
    # ... etc
```

💡 **Tip**: Use absolute paths. You can drag folders into Terminal to get the full path.

### 8. Test the Installation

Open terminal in VS Code (Ctrl+`):

```bash
# Activate environment (if using venv)
source venv/bin/activate

# Test Python imports
python -c "import torch, timm, geopandas; print('All imports successful!')"

# Run help
python tree_crown_pipeline.py --help
```

## Running the Pipeline in VS Code

### Method 1: Integrated Terminal

1. Open terminal: Ctrl+`
2. Activate environment:
   ```bash
   source venv/bin/activate  # or conda activate tree-crown
   ```
3. Run:
   ```bash
   python tree_crown_pipeline.py --step 1
   ```

### Method 2: Run Python File

1. Open `tree_crown_pipeline.py`
2. Click the ▶️ button in top-right
3. Or press `Cmd+Shift+D` and click "Run"

### Method 3: Debug Mode

1. Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Step 1: Clustering",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tree_crown_pipeline.py",
            "args": ["--step", "1"],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Step 2: Species Assignment",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tree_crown_pipeline.py",
            "args": ["--step", "2"],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Step 3: Validation",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tree_crown_pipeline.py",
            "args": ["--step", "3"],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Step 4: KMZ Export",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tree_crown_pipeline.py",
            "args": ["--step", "4"],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

2. Press `F5` to debug

## Common Mac Issues

### Issue: "pip: command not found"

```bash
# Use python3 -m pip instead
python3 -m pip install -r requirements.txt
```

### Issue: "Permission denied"

```bash
# Don't use sudo! Use virtual environment instead
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "xcrun: error" (missing Xcode tools)

```bash
xcode-select --install
```

### Issue: GDAL/Rasterio installation fails

```bash
# Install GDAL first via Homebrew
brew install gdal

# Then install rasterio
pip install rasterio
```

### Issue: Slow processing on Mac

Mac doesn't have CUDA, so it uses CPU. For large datasets:

1. **Reduce batch size**:
   ```python
   BATCH_SIZE = 4  # In Config class
   ```

2. **Use smaller model**:
   ```python
   MODEL_NAME = 'vit_small_patch14_dinov2.lvd142m'
   ```

3. **Process in chunks**: Run Step 1 on subsets of data

### Issue: matplotlib not showing plots

```bash
# Install backend
pip install PyQt5

# Or use non-interactive backend
export MPLBACKEND=Agg
```

## Performance Tips for Mac

### M1/M2/M3 Macs (Apple Silicon)

PyTorch can use Metal Performance Shaders:

```bash
# Install PyTorch with MPS support
pip install torch torchvision
```

The script will automatically use MPS if available:
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

### Intel Macs

1. **Use multiple cores**: scikit-learn automatically uses all cores
2. **Increase memory**: Close other apps
3. **Process overnight**: Large datasets take time on CPU

## Folder Structure After Setup

```
tree-crown-pipeline/
├── venv/                      # Virtual environment
├── tree_crown_pipeline.py     # Main script
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── SETUP_MAC.md              # This file
├── .gitignore                # Git ignore rules
└── .vscode/                  # VS Code settings
    └── launch.json           # Debug configurations
```

## Quick Command Reference

```bash
# Activate environment
source venv/bin/activate

# Run Step 1
python tree_crown_pipeline.py --step 1

# Run Step 2
python tree_crown_pipeline.py --step 2

# Run all steps
python tree_crown_pipeline.py --step all

# Deactivate environment
deactivate

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Recommended VS Code Settings

Create `.vscode/settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": false,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".DS_Store": true
    },
    "python.terminal.activateEnvironment": true
}
```

## Getting Help

- Check README.md for full documentation
- Open an issue on GitHub
- Run with `--help`: `python tree_crown_pipeline.py --help`

## Next Steps

1. ✅ Install dependencies
2. ✅ Configure paths
3. 📂 Prepare your data (orthomosaics + GeoJSONs)
4. ▶️ Run Step 1: `python tree_crown_pipeline.py --step 1`
5. 👀 Browse cluster folders
6. ✏️ Fill species mapping CSV
7. ▶️ Run Step 2-4

Good luck! 🌳
