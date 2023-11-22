# INF554 Data Challenge

## Installation

Using python virtual environment :

```bash
python --version # Check that you have python >=3.11
python -m venv venv

# On MacOS / Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## File structure
```bash
├── .vscode         # VSCode configuration
│
├── initial         # Files from the kaggle challenge
│   ├── test        # Data for test
│   └── training    # Data for training
│
└── src             # Python script dependencies
```