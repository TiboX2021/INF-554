"""
Combine different classifiers and use boosting methods to produce better f1 scores.


Methods yet to be tried :

Boosted classifiers
* RandomForestClassifier
* GradientBoostingClassifier
* AdaBoostClassifier

* BaggingClassifier
* VotingClassifier


Possibilities :    
* majority vote / mean vote
* boost (xgboost, adaboost, ...)
"""

######################################################################################################
#                                         BASIC CLASSIFIERS                                          #
######################################################################################################


def logistic_regression():
    """Returns a logistic regression classifier optimized for this specific task.

    Performance:
    * accuracy : 0.77
    * f1 score : 0.57
    """
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, class_weight="balanced")


def ridge_classifier():
    """Returns a ridge classifier optimized for this specific task.

    Performance:
    * accuracy : 0.76
    * f1 score : 0.57
    """
    from sklearn.linear_model import RidgeClassifier

    return RidgeClassifier(max_iter=1000, class_weight="balanced")


def decision_tree_classifier():
    """Returns a decision tree classifier optimized for this specific task.

    Performance:
    * accuracy : 0.68
    * f1 score : 0.48
    """
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(
        max_depth=5, min_samples_split=5, min_samples_leaf=5, class_weight="balanced"
    )


def support_vector_classifier():
    """Returns a support vector classifier optimized for this specific task.

    NOTE : does not like high dimensionality. The embeddings must be reduced before applying this model...
    """
    from sklearn.svm import LinearSVC

    return LinearSVC(class_weight="balanced")


def nearest_neighbors_classifier():
    """Returns a nearest neighbors classifier optimized for this specific task.

    Performance:
    * accuracy : 0.83
    * f1 score : 0.43
    """
    from sklearn.neighbors import KNeighborsClassifier

    # Overfits
    return KNeighborsClassifier(weights="distance")


######################################################################################################
#                                        BOOSTED CLASSIFIERS                                         #
######################################################################################################


if __name__ == "__main__":
    from loader import get_full_preembedded_training_sets
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    # Load the big bert model
    X_full, y_full = get_full_preembedded_training_sets("all-mpnet-base-v2")
    X_full = X_full.numpy()
    y_full = y_full.numpy().ravel()  # Remove the extra dimension

    print("Using utterances embedded into", X_full.shape[1], "dimensions")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=0
    )

    clf = nearest_neighbors_classifier()

    print("Fitting the classifier...")
    clf.fit(X_train, y_train)
    print("Done.")

    # Test on training
    print("Testing on training data...")
    y_predict_train = clf.predict(X_train)
    print("Accuracy score:", accuracy_score(y_train, y_predict_train))
    print("F1 score:", f1_score(y_train, y_predict_train))
    print("Confusion matrix:")
    print(confusion_matrix(y_train, y_predict_train))

    print("Testing on testing data...")
    y_predict = clf.predict(X_test)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("F1 score:", f1_score(y_test, y_predict))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_predict))
