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

## Usage

### Data

All training data files (`json` utterances and `.txt` graph edges) must be located in the [`data/training/`](./data/training/) directory. The same goes for testing data files in the [`data/test/`](./data/test/) directory.

### Generate utterance text embeddings

We used pretrained embedders from the `sentence_transformers` library.

You can test the performance of our own custom embedder with the following command (we ended up not using it because it would always overfit)

```bash
python src/word_embedding.py
```

In order to save time when embedding utterance, you must first generate the preembedded tensors for each data file. This can be done with the following command :

```bash
python src/loader.py
```

This generates prembedded tensors using 2 models : `all-MiniLM-L6-v2` (small) and `all-mpnet-base-v2` (large).

### Run the classifiers

We experimented on different methods to classify utterances as important or not. The best results were obtained with a simple logistic regression on the embeddings. Our RNN and GNN attempts as well as reimplementations of lab8 and lab9 adapted for this task can be found in the [`rnn.py`](./src/rnn.py), [`graph_model.py`](./src/graph_model.py), [`lab_lstm.py`](./src/lab8_lstm.py) and [`lab9_gnn.py`](./src/lab9_gcn.py).

You can run the logistic regression classifier with the following command :

```bash
python src/majority.py
```

## File structure

```bash
├── .vscode         # VSCode configuration
│
├── data            # Datasets
│   ├── test        # Data for testing
│   └── training    # Data for training
│
├── labels          # Computed labels + true training labels
│   ├── test        # Labels for testing
│   └── training    # Labels for training
│
├── notebooks       # Jupyter notebooks (experiments)
│
└── src             # Python script dependencies
```

## Usage

Run the baseline text-only (no graph) model with the following command.
It will test it against training and validation data, and write to labels/train the estimated test labels.

```bash
python src/baseline.py
```

Use the following command to display help about the function that generates a submission file.

```bash
python src/make_submission.py -h
```
