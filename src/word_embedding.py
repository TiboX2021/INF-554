from typing import Iterable

import numpy as np
import numpy.typing as npt
from loader import get_full_training_sets
from tqdm import tqdm


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
    score_mask = np.logical_or(scores > score_threshold, scores < -score_threshold)
    # score_mask = scores > score_threshold
    chosen_words = filtered_labels[score_mask]
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

        if vector.sum() == 0:
            return vector

        return (vector - vector.mean()) / vector.std()

    def encode_batch(self, strings: Iterable[str], show_progress: bool = True):
        """Encode a batch of strings using the given dictionary."""
        vectors = []

        for string in tqdm(strings, disable=not show_progress):
            vectors.append(self.encode(string))

        return np.array(vectors)

    def size(self) -> int:
        """Return the size of the dictionary, which is the size of the embedded vector space."""
        return len(self.dictionary)


if __name__ == "__main__":
    # Test dictionary generation
    from loader import get_train_test_split_sets
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.tree import DecisionTreeClassifier

    # Load the data
    (
        X_train,
        y_train,
        train_adjacency_matrix,
        X_test,
        y_test,
        test_adjacency_matrix,
    ) = get_train_test_split_sets(0.2)

    # Prepare both text embedders
    words = dictionary_from_data(X_train, y_train, percentile=95, score_threshold=0.6)
    embedder = DictionaryEmbedder(words)
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode all data
    X_train_bert = bert.encode(X_train, show_progress_bar=True)
    X_train_dict = embedder.encode_batch(X_train, show_progress=True)
    X_test_bert = bert.encode(X_test, show_progress_bar=True)
    X_test_dict = embedder.encode_batch(X_test, show_progress=True)

    # Prepare both classifiers. The parameters prevent overfitting
    bert_clf = DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=0
    )
    dict_clf = DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=0
    )

    # Train both classifiers
    print("Training BERT classifier...")
    bert_clf.fit(X_train_bert, y_train)
    print("Done.")

    print("Training dictionary classifier...")
    dict_clf.fit(X_train_dict, y_train)
    print("Done.")

    # Test both classifiers
    print("Testing BERT classifier...")
    bert_train_predict = bert_clf.predict(X_train_bert)
    bert_test_predict = bert_clf.predict(X_test_bert)
    print("Done.")

    print("Testing dictionary classifier...")
    dict_train_predict = dict_clf.predict(X_train_dict)
    dict_test_predict = dict_clf.predict(X_test_dict)
    print("Done.")

    # Compare scores
    print("BERT training score :", accuracy_score(y_train, bert_train_predict))
    print("BERT testing score :", accuracy_score(y_test, bert_test_predict))
    print("Dictionary training score :", accuracy_score(y_train, dict_train_predict))
    print("Dictionary testing score :", accuracy_score(y_test, dict_test_predict))

    # Compare confusion matrices
    print("BERT confusion matrix :")
    print(confusion_matrix(y_test, bert_test_predict))
    print("Dictionary confusion matrix :")
    print(confusion_matrix(y_test, dict_test_predict))

    # Compare F1 scores
    print("BERT F1 score :", f1_score(y_test, bert_test_predict))
    print("Dictionary F1 score :", f1_score(y_test, dict_test_predict))
