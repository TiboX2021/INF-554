from typing import Iterable

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn, optim
from visualize import detach_tensor, plot_2D_embeddings

############################################################################################################
#                                      TORCH CONVERSION FUNCTIONS                                          #
############################################################################################################


def build_adjacency_from_edges(edges_iterable: Iterable[tuple[str, np.ndarray]]):
    """Build an adjacency matrix from a list of edge datasets"""
    node_count = 0
    total_edges = np.array([], dtype=np.int32).reshape(0, 2)

    for _, edges in edges_iterable:
        # Offset the node ids in relation to the previous subgraphs
        max_index = edges.max() + 1
        edges += node_count
        node_count += max_index

        # Add the edges to the total edges array
        total_edges = np.concatenate((total_edges, edges))  # type: ignore

    # Create the adjacency matrix
    adjacency_matrix = sp.coo_matrix(
        (np.ones(total_edges.shape[0]), (total_edges[:, 0], total_edges[:, 1])),
        shape=(node_count, node_count),
        dtype=np.float32,
    )

    return adjacency_matrix


def normalise_adjacency(matrix: sp.coo_matrix) -> sp.csr_matrix:
    """Normalize and preprocess a graph adjacency matrix for use in a GCN model"""
    n = matrix.shape[0]
    diag = matrix.sum(0)
    diag = 1 / (np.sqrt(diag + 1))
    sparse_diag = sp.diags(np.squeeze(np.asarray(diag)))
    normalized_matrix = sparse_diag @ (matrix + sp.eye(n)) @ sparse_diag

    return normalized_matrix


def csr_to_coo_matrix(matrix: sp.csr_matrix) -> sp.coo_matrix:
    return matrix.tocoo().astype(np.float32)  # type: ignore


def sparse_to_torch_sparse(matrix: sp.coo_matrix) -> torch.Tensor:
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""

    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    values = torch.from_numpy(matrix.data)
    shape = torch.Size(matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape)  # type: ignore


def full_adjacency_preprocessing(matrix: sp.coo_matrix):
    csr_matrix = normalise_adjacency(matrix)
    csr_matrix = csr_to_coo_matrix(csr_matrix)
    tensor_matrix = sparse_to_torch_sparse(csr_matrix)
    return tensor_matrix


############################################################################################################
#                                          TORCH MODEL UTILITIES                                           #
############################################################################################################


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
):
    """Trains a Torch model.
    This function expects that the `forward` method of the model returns a tuple of two elements:
    - the output of the model
    - the embeddings of the hidden layer

    Params:
        - model (nn.Module) : the model to train.
        - optimizer (optim.Optimizer) : the optimizer to use for training.
        - loss_function (nn.Module) : the loss function to use for training.
        - X_train (torch.Tensor) : the training data.
        - y_train (torch.Tensor) : the training labels.
        - epochs (int) : the number of epochs to train the model for.
    """
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output, _embeddings = model(X_train)

        # Loss computation
        loss = loss_function(output, y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print loss once every 10 epochs
        if epoch % 10 == 0:
            # Evaluate accuracy and f1_score on the current training batch
            accuracy = accuracy_score(y_train.cpu(), output.cpu().ge(0.5))
            f1 = f1_score(y_train.cpu(), output.cpu().ge(0.5))

            print(
                f"Epoch {epoch} : loss = {loss.item():.2f} | accuracy = {accuracy:.2f} | f1_score = {f1:.2f}"
            )


def test_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    show_embeddings: bool = True,
):
    """Test a Torch Model and display useful metrics :
    - accuracy
    - f1 score
    - confusion matrix

    Params:
        - model (nn.Module) : the model to test.
        - X_test (torch.Tensor) : the test data.
        - y_test (torch.Tensor) : the test labels.
        - show_embeddings (bool) : whether to display the embeddings or not, using matplotlib.
    """

    # Sets model in evaluation mode (disable dropout, etc)
    model.eval()

    # Forward pass
    output, _embeddings = model(X_test)

    # Detach tensors and convert them to numpy arrays
    detach_output = detach_tensor(output.ge(0.5)).reshape(-1)
    detach_y_test = detach_tensor(y_test).reshape(-1)

    # Evaluate metrics
    accuracy = accuracy_score(detach_y_test, detach_output)
    f1 = f1_score(detach_y_test, detach_output)
    cm = confusion_matrix(detach_y_test, detach_output)

    # Print metrics
    print(f"Accuracy : {accuracy:.2f}")
    print(f"F1 score : {f1:.2f}")
    print("Confusion matrix :")
    print(cm)

    # Plot embeddings
    if show_embeddings:
        plot_2D_embeddings(detach_tensor(_embeddings), detach_y_test)
