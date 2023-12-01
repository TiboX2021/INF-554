import json
from pathlib import Path

from loader import get_full_training_sets, get_test_data_iterator
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#############################################################################################
# text_baseline: utterances are embedded with SentenceTransformer, then train a classifier. #
#############################################################################################

# XXX sentence-transformer is a state-of-the-art text embedder package
# This one (all-MiniLM-L6-v2) embeds each string into 384 float32 dimensions
# TODO : if we reuse the same embedding, precompute it into a txt file and load it via loadfromtxt + torch tensors
# TODO : faire des fonctions pour ça dans loader ! Un write, un load.
bert = SentenceTransformer("all-MiniLM-L6-v2")

# Extract all training data
X_training, y_training = get_full_training_sets()

# Compute the bert embedding for each utterance
X_training = bert.encode(X_training, show_progress_bar=True)

# In order to test the models for overfitting, we need to split the training data into training and validation sets
X_training, X_validation, y_training, y_validation = train_test_split(
    X_training,
    y_training,
    test_size=0.2,
    random_state=0,  # random_state=0 for reproducibility
)


# XXX : use a basic classifier from this encoded data
print("Training classifier...")
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_training, y_training)
print("Done.")

#############################################################################################
#                                     TESTING AND VALIDATION                                #
#############################################################################################

# Testing on the training data
print("Testing on training data...")
y_pred = clf.predict(X_training)
score = clf.score(X_training, y_training)
print("Score on the 80% training set:", score)

# Testing on the validation data
print("Testing on validation data... (overfitting check)")
y_pred = clf.predict(X_validation)
score = clf.score(X_validation, y_validation)
print("Score on the 20% validation set:", score)

# Producing test labels file
print("Producing test labels file...")
test_labels = {}
for transcription_id, utterances in get_test_data_iterator():
    X_test = []
    for utterance in utterances:
        X_test.append(f"{utterance['speaker']}: {utterance['text']}")

    X_test = bert.encode(X_test)

    y_test = clf.predict(X_test)
    test_labels[transcription_id] = y_test.tolist()

print("Writing test labels file...")
with open(
    Path("labels") / Path("test") / "test_labels_text_baseline.json", "w"
) as file:
    json.dump(test_labels, file, indent=4)
