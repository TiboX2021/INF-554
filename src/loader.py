import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
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


# TODO : also allow to read edge labels for a more performant model ?


if __name__ == "__main__":
    # Print out the sets
    print(f"Training set: ({len(training_set)}) :", training_set)
    print()
    print(f"Test set: ({len(test_set)})", test_set)
