import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from utils import build_adjacency_from_edges


class Utterance(TypedDict):
    """An utterance from a transcription dataset"""

    speaker: str
    text: str
    index: int


def flatten(list_of_list: list[list]):
    """Flatten a list of lists into a single list"""
    return [item for sublist in list_of_list for item in sublist]


# Paths
data_folder = Path("data")
labels_folder = Path("labels")

training_data_path = data_folder / Path("training")
testing_data_path = data_folder / Path("test")
training_labels_path = labels_folder / Path("training")
testing_labels_path = labels_folder / Path("test")

############################################################################################################
#                                              DATASET FILENAMES                                           #
############################################################################################################


#####
# training and test sets of transcription ids
#####
training_set = [
    "ES2002",
    "ES2005",
    "ES2006",
    "ES2007",
    "ES2008",
    "ES2009",
    "ES2010",
    "ES2012",
    "ES2013",
    "ES2015",
    "ES2016",
    "IS1000",
    "IS1001",
    "IS1002",
    "IS1003",
    "IS1004",
    "IS1005",
    "IS1006",
    "IS1007",
    "TS3005",
    "TS3008",
    "TS3009",
    "TS3010",
    "TS3011",
    "TS3012",
]


test_set = [
    "ES2003",
    "ES2004",
    "ES2011",
    "ES2014",
    "IS1008",
    "IS1009",
    "TS3003",
    "TS3004",
    "TS3006",
    "TS3007",
]

# Data files basenames
training_set = flatten([[m_id + s_id for s_id in "abcd"] for m_id in training_set])
test_set = flatten([[m_id + s_id for s_id in "abcd"] for m_id in test_set])

# Remove names that are absent from the training set
training_set.remove("IS1002a")
training_set.remove("IS1005d")
training_set.remove("TS3012c")


EDGE_LABELS = [
    "Continuation",
    "Explanation",
    "Elaboration",
    "Acknowledgement",
    "Comment",
    "Result",
    "Question-answer_pair",
    "Contrast",
    "Clarification_question",
    "Background",
    "Narration",
    "Alternation",
    "Conditional",
    "Q-Elab",
    "Correction",
    "Parallel",
]

EDGE_LOOKUP = {label: index for index, label in enumerate(EDGE_LABELS)}


def edge_label_to_index(label: str):
    """Converts an edge label to its index in the EDGE_LABELS list"""
    return EDGE_LOOKUP[label]


np_edge_label_to_index = np.vectorize(edge_label_to_index)


def index_to_edge_label(index: int):
    """Converts an edge label index to its string representation"""
    return EDGE_LABELS[index]


np_index_to_edge_label = np.vectorize(index_to_edge_label)

############################################################################################################
#                                              DATASET ITERATORS                                           #
############################################################################################################


def get_training_data_iterator():
    """Returns an iterator over the training datasets data, that can be iterated over using a for loop"""
    for transcription_id in training_set:
        with open(training_data_path / f"{transcription_id}.json", "r") as file:
            data: list[Utterance] = json.load(file)
            yield transcription_id, data


def get_test_data_iterator():
    """Returns an iterator over the test datasets data, that can be iterated over using a for loop"""
    for transcription_id in test_set:
        with open(testing_data_path / f"{transcription_id}.json", "r") as file:
            data: list[Utterance] = json.load(file)
            yield transcription_id, data


def get_training_edges_iterator():
    """Returns an iterator over the training datasets data, that can be iterated over using a for loop

    Dataset specificity :
        - The nodes are already numbered from 0 to n_nodes - 1
        - All nodes in a subgraph are connected. We can get the total node count with max(node_ids) + 1
    """
    for transcription_id in training_set:
        with open(training_data_path / f"{transcription_id}.txt", "r") as file:
            edges = np.genfromtxt(file, dtype=np.int32)

            yield transcription_id, edges[:, [0, 2]]


def get_test_edges_iterator():
    """Returns an iterator over the test datasets data, that can be iterated over using a for loop

    Dataset specificity :
        - The nodes are already numbered from 0 to n_nodes - 1
        - All nodes in a subgraph are connected. We can get the total node count with max(node_ids) + 1
    """
    for transcription_id in test_set:
        with open(testing_data_path / f"{transcription_id}.txt", "r") as file:
            edges = np.genfromtxt(file, dtype=np.int32)

            yield transcription_id, edges[:, [0, 2]]


############################################################################################################
#                                            DATASET FULL LOADERS                                          #
############################################################################################################


def get_full_training_sets():
    """Aggregates all training data and labels. Only useful for basic text-only models (like the baseline text one)

    Returns
        - X_train : list[str] - A list of all utterances from every training transcription.
        - y_train : list[int] - A list of all labels (1 = mportant | 0 = not important) from every training transcription.
    """
    X_train: list[str] = []
    y_train: list[int] = []

    # Iterate over training labels, and fill X_train, y_train in the same order
    with open(training_labels_path / "training_labels.json", "r") as file:
        training_labels = json.load(file)

        # Iterate over training data and fill X_train
        for transcription_id, transcription in get_training_data_iterator():
            for utterance in transcription:
                # Agregate the speaker ID into the utterance text (cf baseline example)
                X_train.append(f"{utterance['speaker']} {utterance['text']}")
            y_train.extend(training_labels[transcription_id])

    return X_train, y_train


def get_full_preembedded_training_sets(bert_embedder: str):
    """Load all training data and labels, but uses the precomputed global training embeddings instead of loading the raw text data"""
    from torch import FloatTensor

    try:
        # Load the precomputed embeddings
        X_train: Tensor = torch.load(
            training_data_path / Path(f"training_{bert_embedder}.pth")
        )

        # Load the labels
        with open(training_labels_path / "training_labels.json", "r") as file:
            training_labels = json.load(file)

        # Build the training sets in the order of the training_set list
        y_train = []
        for transcription_id in training_set:
            y_train.extend(training_labels[transcription_id])

        return FloatTensor(X_train), FloatTensor(y_train).view(-1, 1)

    except FileNotFoundError:
        print(
            "You need to build the precomputed embeddings first. Run build_all_pth() from src/loader.py"
        )
        exit(1)


def get_full_preembedded_test_sets(bert_embedder: str):
    """Load all test data, but uses the precomputed global test embeddings instead of loading the raw text data.

    NOTE : because we need to associate the labels with each test file, this function returns a list of tuples :
    list[(transcription_id, embeddings)]
    """
    from torch import FloatTensor

    embeddings: list[tuple[str, Tensor]] = []

    for transcription_id in test_set:
        tensor = torch.load(
            testing_data_path / Path(f"{transcription_id}_{bert_embedder}.pth")
        )
        embeddings.append((transcription_id, FloatTensor(tensor)))

    return embeddings


def get_train_test_split_sets(test_size: float = 0.2, random_state: int | None = None):
    """Returns a training dataset and a testing dataset taken from the original training dataset,
    in order to be able to test with labels that we have.

    The split is not performed randomly on the agregated data, but on the file names.
    This way, we preserve each subdataset's graph integrity
    """
    from sklearn.model_selection import train_test_split

    # Perform a random splitting of the training set file names
    train_files, test_files = train_test_split(
        training_set, test_size=test_size, random_state=random_state
    )

    # Build individual adjacency matrixes and label sets
    train_edges: list[tuple[str, np.ndarray]] = []
    test_edges: list[tuple[str, np.ndarray]] = []

    X_train: list[str] = []
    X_test: list[str] = []

    y_train: list[int] = []
    y_test: list[int] = []

    # Load all training labels
    with open(training_labels_path / "training_labels.json", "r") as file:
        training_labels = json.load(file)

    # Create the training data
    for transcription_id in train_files:
        # Build text and labels
        with open(training_data_path / f"{transcription_id}.json", "r") as file:
            transcription: list[Utterance] = json.load(file)
            for utterance in transcription:
                # Agregate the speaker ID into the utterance text (cf baseline example)
                X_train.append(f"{utterance['speaker']}: {utterance['text']}")
            y_train.extend(training_labels[transcription_id])

        # Build edges
        with open(training_data_path / f"{transcription_id}.txt", "r") as file:
            edges = np.genfromtxt(file, dtype=np.int32)
            train_edges.append((transcription_id, edges[:, [0, 2]]))

    # Create the testing data
    for transcription_id in test_files:
        with open(training_data_path / f"{transcription_id}.json", "r") as file:
            transcription: list[Utterance] = json.load(file)
            for utterance in transcription:
                # Agregate the speaker ID into the utterance text (cf baseline example)
                X_test.append(f"{utterance['speaker']}: {utterance['text']}")
            y_test.extend(training_labels[transcription_id])

        # Build edges
        with open(training_data_path / f"{transcription_id}.txt", "r") as file:
            edges = np.genfromtxt(file, dtype=np.int32)
            test_edges.append((transcription_id, edges[:, [0, 2]]))

    train_adjacency_matrix = build_adjacency_from_edges(train_edges)
    test_adjacency_matrix = build_adjacency_from_edges(test_edges)

    return (
        X_train,
        y_train,
        train_adjacency_matrix,
        X_test,
        y_test,
        test_adjacency_matrix,
    )


def get_training_adjacency_matrix():
    """Returns a sparse adjacency matrix for the full training dataset"""
    return build_adjacency_from_edges(get_training_edges_iterator())


def get_test_adjacency_matrix():
    """Returns a sparse adjacency matrix for the full test dataset"""
    return build_adjacency_from_edges(get_test_edges_iterator())


def get_device():
    """Get torch device"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


############################################################################################################
#                                          TORCH DATA UTILITIES                                            #
############################################################################################################


def build_all_pth(bert_embedder: str):
    """Builds all the .pth files using a BERT embedder for each training and testing set.
    Also builds the global .pth file for all words from the testing set and training set.

    The idea is to avoid having to rerun the embedder each time we want to train a model.
    """
    print(f"Building .pth files for {bert_embedder} BERT embedder...")

    from sentence_transformers import SentenceTransformer
    from torch import Tensor

    bert = SentenceTransformer(bert_embedder)

    global_tensor = torch.Tensor()

    # Compute embeddings for the training set
    for transcription_id, data in get_training_data_iterator():
        sentences = [
            f"{utterance['speaker']} {utterance['text']}" for utterance in data
        ]

        embedded_utterances = Tensor(bert.encode(sentences, show_progress_bar=True))

        # Save the embeddings in a .pth file
        torch.save(
            embedded_utterances,
            training_data_path / Path(f"{transcription_id}_{bert_embedder}.pth"),
        )
        print(f"Saved {transcription_id}_{bert_embedder}.pth")

        # Append the embeddings to the global tensor
        global_tensor = torch.cat((global_tensor, embedded_utterances), dim=0)

    # Save the global training tensor
    torch.save(
        global_tensor,
        training_data_path / Path(f"training_{bert_embedder}.pth"),
    )

    # Reset the global tensor
    global_tensor = torch.Tensor()

    # Compute embeddings for the testing set
    for transcription_id, data in get_test_data_iterator():
        sentences = [
            f"{utterance['speaker']} {utterance['text']}" for utterance in data
        ]

        embedded_utterances = Tensor(bert.encode(sentences, show_progress_bar=True))

        # Save the embeddings in a .pth file
        torch.save(
            embedded_utterances,
            testing_data_path / Path(f"{transcription_id}_{bert_embedder}.pth"),
        )
        print(f"Saved {transcription_id}_{bert_embedder}.pth")

        # Append the embeddings to the global tensor
        global_tensor = torch.cat((global_tensor, embedded_utterances), dim=0)

    # Save the global testing tensor
    torch.save(
        global_tensor,
        testing_data_path / Path(f"test_{bert_embedder}.pth"),
    )


############################################################################################################
#                             torch_geometric GRAPH LOADING FUNCTIONS                                      #
############################################################################################################


def geometric_train_test_split(
    bert: str, test_size: float = 0.2, random_state: int | None = None
):
    """Loads training and testing data for the torch_geometric graph model.

    Returns:
        - X_train : torch.Tensor - A tensor of all precomputed embeddings from every training transcription.
        - y_train : torch.Tensor - A tensor of all labels (1 = mportant | 0 = not important) from every training transcription.
        - train_edges : np.ndarray - A numpy array of all edges from every training transcription.
        - X_test : torch.Tensor - A tensor of all precomputed embeddings from every testing transcription.
        - y_test : torch.Tensor - A tensor of all labels (1 = mportant | 0 = not important) from every testing transcription.
        - test_edges : np.ndarray - A numpy array of all edges from every testing transcription.
    """
    from sklearn.model_selection import train_test_split

    # Perform a random splitting of the training set file names
    train_files, test_files = train_test_split(
        training_set, test_size=test_size, random_state=random_state
    )

    # Build individual adjacency matrixes and label sets
    train_edges = np.empty((0, 3))
    test_edges = np.empty((0, 3))

    # Prepare preembedded data & train labels already in Tensors (ready for use in RNN)
    X_train = Tensor()
    X_test = Tensor()

    y_train = Tensor()
    y_test = Tensor()

    node_count = 0  # Offset the node indices for each new subgraph that we concatenate

    # Load all training labels
    with open(training_labels_path / "training_labels.json", "r") as file:
        training_labels = json.load(file)

    # Create the training data
    for transcription_id in train_files:
        # Load the precomputed embeddings tensor
        preembedded_utterances = torch.load(
            training_data_path / Path(f"{transcription_id}_{bert}.pth")
        )
        X_train = torch.cat((X_train, preembedded_utterances), dim=0)

        # Load the labels
        y_train = torch.cat(
            (y_train, Tensor(training_labels[transcription_id]).view(-1, 1)), dim=0
        )

        # Load the edges
        with open(training_data_path / f"{transcription_id}.txt", "r") as file:
            edges = np.genfromtxt(file, dtype=str)
            # Extract the node indices and offset them
            int_edges = edges[:, [0, 2]].astype(np.int32) + node_count

            # Reconcatenate everything into a str numpy array
            # Extract the columns from int_edges and edges
            first_column = int_edges[:, 0]
            middle_column = edges[:, 1]
            second_column = int_edges[:, 1]

            # Recreate the str numpy array
            recreated_array = np.column_stack(
                (first_column, middle_column, second_column)
            ).astype(str)
            train_edges = np.concatenate((train_edges, recreated_array), axis=0)

            # Update the node count
            node_count = X_train.shape[0]

    # Reset the node count for the testing set
    node_count = 0

    # Create the training data
    for transcription_id in test_files:
        # Load the precomputed embeddings tensor
        preembedded_utterances = torch.load(
            training_data_path / Path(f"{transcription_id}_{bert}.pth")
        )
        X_test = torch.cat((X_test, preembedded_utterances), dim=0)

        # Load the labels
        y_test = torch.cat(
            (y_test, Tensor(training_labels[transcription_id]).view(-1, 1)), dim=0
        )

        # Load the edges
        with open(training_data_path / f"{transcription_id}.txt", "r") as file:
            edges = np.genfromtxt(file, dtype=str)
            # Extract the node indices and offset them
            int_edges = edges[:, [0, 2]].astype(np.int32) + node_count

            # Reconcatenate everything into a str numpy array
            # Extract the columns from int_edges and edges
            first_column = int_edges[:, 0]
            middle_column = edges[:, 1]
            second_column = int_edges[:, 1]

            # Recreate the str numpy array
            recreated_array = np.column_stack(
                (first_column, middle_column, second_column)
            ).astype(str)
            test_edges = np.concatenate((test_edges, recreated_array), axis=0)

            # Update the node count
            node_count = X_test.shape[0]

    # Return the data
    return (
        X_train,
        y_train,
        train_edges,
        X_test,
        y_test,
        test_edges,
    )


if __name__ == "__main__":
    # Prebuild the embeddings for the training and testing sets
    lil_bert = "all-MiniLM-L6-v2"
    fat_bert = "all-mpnet-base-v2"

    build_all_pth(lil_bert)
    build_all_pth(fat_bert)
