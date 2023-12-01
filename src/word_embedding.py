import numpy as np
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


# TODO : create function to generate a training dictionary from training data

# TODO : test the embedder performance


class DictionaryEmbedder:
    pass


if __name__ == "__main__":
    pass
    # word_counts = words_label_count()

    # with open("word_counts.json", "w") as file:
    #     json.dump(word_counts, file, indent=2)
