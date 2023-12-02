import torch
import torch.nn.functional as F
import torch.optim as optim
from loader import (
    get_device,
    get_train_test_split_sets,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from utils import full_adjacency_preprocessing
from visualize import detach_tensor, plot_2D_embeddings


class Lab9_GCN(nn.Module):
    """GCN model based on the lab 9 in order to take advantage of text AND graph features of the dataset.

    This model uses text and graph data for training, but only takes text into account for evaluation.
    One possible optimization would be to try using a combination of models to be able to exploit graph data for evaluation also.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden_1: int,
        n_hidden_2: int,
        n_class: int,
        dropout: float,
    ):
        """
        Initialize the Lab9_GCN model

        Parameters :
            - n_features (int) : the number of features of the input data. This number corresponds to the dimension
              of the embedded data (384 if using the bert all-MiniLM-L6-v2 text embedder).
            - n_hidden_1 (int) : the number of neurons for the first hidden layer.
            - n_hidden_2 (int) : the number of neurons for the second hidden layer. This is the last hidden layer,
              which corresponds to the model's internal embeddings for text.
            - n_class (int) : the number of output classes. For binary classification, this parameter must be set to 1.
        """
        super(Lab9_GCN, self).__init__()

        # 3 couches de multiplication matricielle des features avec la matrice de poids
        self.fc1 = nn.Linear(n_features, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)

        # TODO : use weight decay instead of dropout
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor):
        z0 = self.fc1(x)
        z0 = self.relu(torch.mm(adjacency_matrix, z0))
        z0 = self.dropout(z0)

        z2 = self.fc2(z0)
        z2 = self.relu(z2)

        x = self.fc3(z2)

        # Sigmoid function for binary classification
        return F.sigmoid(x), z2


if __name__ == "__main__":
    device = get_device()
    print("Using device", device)

    # Load training graph datasets
    (
        X_train,
        y_train,
        train_adjacency_matrix,
        X_test,
        y_test,
        test_adjacency_matrix,
    ) = get_train_test_split_sets(0.2, 0)

    # Preprocess matrixes
    train_adjacency_matrix = full_adjacency_preprocessing(train_adjacency_matrix).to(
        device
    )
    test_adjacency_matrix = full_adjacency_preprocessing(test_adjacency_matrix).to(
        device
    )

    # Embed the words
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = bert.encode(X_train, show_progress_bar=True)
    X_test = bert.encode(X_test, show_progress_bar=True)

    # Move encoded sentences to float torch tensors to gpu
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)

    # Move labels to float tensors + add a secondary acix for loss computing
    y_train = torch.FloatTensor(y_train).to(device).view(-1, 1)
    y_test = torch.FloatTensor(y_test).to(device).view(-1, 1)

    # Instanciate model
    model = Lab9_GCN(384, 256, 32, 1, 0.5).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.BCELoss()

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        output, _ = model(X_train, train_adjacency_matrix)

        loss = loss_function(output, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate performance on training data
        predicted = output.ge(0.5).type(torch.FloatTensor)
        accuracy = (predicted.cpu() == y_train.cpu()).sum().item() / output.size()[0]
        print("Train accuracy for epoch", epoch, "is", round(accuracy, 2))

    # Set model in evaluation mode (disable dropout)
    model.eval()

    # Final test on training data
    train_output, train_embeddings = model(X_train, train_adjacency_matrix)
    predicted = train_output.ge(0.5).type(torch.FloatTensor)
    accuracy = (predicted.cpu() == y_train.cpu()).sum().item() / train_output.size()[0]
    print("Accuracy on train data is", round(accuracy, 2))

    # Test on validation data
    test_output, test_embeddings = model(X_test, test_adjacency_matrix)
    predicted = test_output.ge(0.5).type(torch.FloatTensor)
    accuracy = (predicted.cpu() == y_test.cpu()).sum().item() / test_output.size()[0]
    print("Accuracy on test data is", round(accuracy, 2))

    # Display confusion matrixes
    np_y_train = detach_tensor(y_train).reshape(-1)
    np_y_test = detach_tensor(y_test).reshape(-1)
    np_train_output = detach_tensor(train_output.ge(0.5)).reshape(-1)
    np_test_output = detach_tensor(test_output.ge(0.5)).reshape(-1)

    # Confusion matrix on training data
    print("Confusion matrix on training data")
    train_confusion = confusion_matrix(np_y_train, np_train_output)
    print(train_confusion)

    # Confusion matrix on test data
    print("Confusion matrix on test data")
    test_confusion = confusion_matrix(np_y_test, np_test_output)
    print(test_confusion)

    print("F1 score on training data :", f1_score(np_y_train, np_train_output))
    print("F1 score on test data :", f1_score(np_y_test, np_test_output))

    # Print embeddings
    print("Plotting embeddings...")
    plot_2D_embeddings(
        detach_tensor(train_embeddings),
        detach_tensor(y_train).reshape(-1),
    )
    plot_2D_embeddings(
        detach_tensor(test_embeddings),
        detach_tensor(y_test).reshape(-1),
    )

    print("Done !")
