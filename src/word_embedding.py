from typing import Iterable

import numpy as np
import numpy.typing as npt
from loader import get_full_training_sets
from tqdm import tqdm

# Ratio of positive utterances (for normalization of word occurrences)
POSITIVE_RATIO = 0.1830274


def words_label_count():
    """Count the occurrence of each word in utterances of each label.

    Returns:
        dict[str, [int, int]]: A dictionary with words as keys. To each word is associated a list of 2 counts
        list[0] =  number of times the word appears in utterances of label 0
        list[1] =  number of times the word appears in utterances of label 1
    """

    X_train, y_train = get_full_training_sets()

    word_counts: dict[str, list[int]] = {}
    n = len(X_train)

    positive_utterances = 0

    for i, (utterance, label) in enumerate(zip(X_train, y_train)):
        positive_utterances += label

        words = np.char.lower(utterance.split(sep=" "))

        print(f"\rUtterance {i+1} of - {n}", end="", flush=True)

        for word in words:
            if word not in word_counts:
                word_counts[word] = [0, 0]

            word_counts[word][label] += 1
    print("\nDone")

    print(
        f"Positive utterances : {positive_utterances} / {n} ({positive_utterances / n * 100:.5f}%)"
    )

    return word_counts


def dictionary_from_data(
    utterances: list[str],
    labels: list[int],
    percentile: int = 95,
    score_threshold: float = 0.5,
) -> npt.NDArray[np.str_]:
    """Creates a best-words dictionary from a list of utterances and their labels.
    This allows using dictionaries generated from training data only and not testing data,
    which represents better the Kaggle situation.

    Params:
        - utterances (list[str]) : A list of utterances strings.
        - labels (list[int]) : A list of labels (1 = important | 0 = not important).
        - percentile (int) : The percentile of words to keep in the dictionary. Defaults to 95 (keep the 5% best).
        - score_threshold (float) : The score threshold to keep a word in the dictionary, in [-1, 1]. Defaults to 0.5.

    Returns:
        A numpy array of strings corresponding to the chosen dictionary.
    """

    # Build the count directory
    word_counts: dict[str, list[int]] = {}
    n = len(utterances)
    positive_ratio = 0

    for i, (utterance, label) in enumerate(zip(utterances, labels)):
        positive_ratio += label

        # Split the utterance words and convert them to lowercase
        print(f"\rUtterance {i+1} of - {n}", end="", flush=True)
        words = np.char.lower(utterance.split(sep=" "))

        for word in words:
            if word not in word_counts:
                word_counts[word] = [0, 0]

            word_counts[word][label] += 1

    positive_ratio /= n

    print()
    print("Positive utterances ratio :", positive_ratio)
    print("Found", len(word_counts), "unique words")

    # Do the word analysis (cf word_embedding.pynb)
    # Build numpy representations
    counts = np.array(list(word_counts.values()), dtype=np.float32)
    occurrences = counts.sum(axis=1)
    counts[:, 0] *= positive_ratio  # Normalize the negative counts
    np_labels = np.array(list(word_counts.keys()))

    # Percentile filtering
    occurence_threshold = np.percentile(occurrences, percentile)
    mask = occurrences > occurence_threshold
    filtered_counts = counts[mask]
    filtered_labels = np_labels[mask]
    print(
        "Kept",
        len(filtered_labels),
        "words of occurrence greater than",
        occurence_threshold,
        " - ",
        percentile,
        "th percentile",
    )

    # Score filtering
    scores = (filtered_counts[:, 1] - filtered_counts[:, 0]) / (
        filtered_counts[:, 1] + filtered_counts[:, 0]
    )
    chosen_words = filtered_labels[scores > score_threshold]
    print("Final dictionary size :", len(chosen_words))

    return chosen_words


class DictionaryEmbedder:
    """A custom embedder that uses a dictionary to embed utterances into a vector space.
    The vector space if of size DICTIONARY_SIZE. Utterances are embedded by counting the occurrences of each word in the dictionary.

    Usage:

    embedder = DictionaryEmbedder(dictionary)
    vector = embedder.encode("This is an utterance")
    vectors = embedder.encode_batch(["This is an utterance", "This is another one"])
    """

    dictionary: npt.NDArray[np.str_]
    dictionary_index_lookup: dict[str, int]

    def __init__(self, dictionary: npt.NDArray[np.str_] | str):
        """Create a dictionary embedder

        Params:
            - dictionary : a numpy array of strings (dictionary) or the path to a text file containing the dictionary
        """
        if isinstance(dictionary, str):
            # Load the dictionary from the file
            with open(dictionary, "r") as f:
                self.dictionary = np.genfromtxt(f, dtype=np.str_)
        else:
            self.dictionary = dictionary

        self.dictionary_index_lookup = {
            word: i for i, word in enumerate(self.dictionary)
        }

    def encode(self, string: str):
        """Encode a string using the given dictionary."""

        vector = np.zeros(len(self.dictionary), dtype=np.float32)

        words = np.char.lower(string.split(sep=" "))
        for word in words:
            if word in self.dictionary_index_lookup:
                vector[self.dictionary_index_lookup[word]] += 1

        return vector

    def encode_batch(self, strings: Iterable[str], show_progress: bool = True):
        """Encode a batch of strings using the given dictionary."""
        vectors = []

        for string in tqdm(strings, disable=not show_progress):
            vectors.append(self.encode(string))

        return vectors

    def size(self) -> int:
        """Return the size of the dictionary, which is the size of the embedded vector space."""
        return len(self.dictionary)


if __name__ == "__main__":
    # Test dictionary generation
    from loader import get_train_test_split_sets

    (
        X_train,
        y_train,
        train_adjacency_matrix,
        X_test,
        y_test,
        test_adjacency_matrix,
    ) = get_train_test_split_sets(0.2, 0)

    words = dictionary_from_data(X_train, y_train)

    embedder = DictionaryEmbedder(words)
    print("Embedder size :", embedder.size())

    print("Embedding", len(X_train), "utterances...")
    result = embedder.encode_batch(X_train)
    print("Done.")
