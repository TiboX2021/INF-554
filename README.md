# INF554 Data Challenge

https://www.kaggle.com/competitions/inf554-extractive-summarization-2023/data

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
├── data            # Datasets
│   ├── test        # Data for testing
│   └── training    # Data for training
│
├── initial         # Some files
│
├── labels          # Computed labels + true training labels
│   ├── test        # Labels for testing
│   └── training    # Labels for training
│
└── src             # Python script dependencies
```

## Usage

Run the baseline text-only (no graph) model with the following command.
It will test it against training and validation data, and write to labels/train the estimated test labels.

```bash
python src/baseline.py
```
