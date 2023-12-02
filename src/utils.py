from typing import Iterable

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim

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
                f"Epoch {epoch} : loss = {loss.item()} | accuracy = {accuracy} | f1_score = {f1}"
            )
