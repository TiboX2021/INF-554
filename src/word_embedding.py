import numpy as np
import numpy.typing as npt
from loader import get_full_training_sets

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
    print("Found", len(word_counts), "unique words")

    # Do the word analysis (cf word_embedding.pynb)
    # Build numpy representations
    counts = np.array(list(word_counts.values()), dtype=np.float32)
    occurrences = counts.sum(axis=1)
    counts[:, 0] /= occurrences  # Normalize the negative counts
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


# TODO : test the embedder performance


class DictionaryEmbedder:
    pass


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

    print(words)
