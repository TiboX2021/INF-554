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


def build_3d_adjacency_from_edges(edges_iterable: Iterable[tuple[str, np.ndarray]]):
    """Build a 3D adjacency matrix from a list of edge datasets.

    Params:
        - edges_iterable (Iterable[tuple[str, np.ndarray]]) : an iterable of tuples containing the name of the graph and the edges + labels of the graph in str format

    Edge labels are encoded in the 3D dimensions. As there are 16 different edge labels, we use 32 dimensions.
        - 1 to 16 encore forward edges
        - 17 to 32 encode backward edges

    A NN processing can be done on the matrix in order to aggregate the 32 dimensions into 1, for classic adjacency multiplication later-on.

    TODO : NN matrix preprocessing : see how to replicate the same thing.

    NOTE : à vérifier pour les ops de multiplication + préprocessing
        - The superior triangle of the matrix is used to encode backward edges.
        - The inferior triangle of the matrix is used to encode forward edges.
    """

    total_edge_count = 0
    for edges in edges_iterable:
        total_edge_count += edges[1].shape[0]

    # Initialize sparse matrix data
    data = np.ones(total_edge_count * 2, dtype=np.int32)

    # We will fill the following arrays with the coordinates
    rows = torch.zeros(total_edge_count * 2)
    cols = torch.zeros(total_edge_count * 2)
    depths = torch.zeros(total_edge_count * 2)

    labels = [
        "Continuation",
        "Explanation",
        "Elaboration",
        "Acknowledgement",
        "Comment",
        "Result",
        "Question-answer_pair",
        "Contrast",
        "Clarification_question",
        "Background",
        "Narration",
        "Alternation",
        "Conditional",
        "Q-Elab",
        "Correction",
        "Parallel",
    ]
    label_lookup = {label: i for i, label in enumerate(labels)}

    node_count = 0  # Because each edge ndarray starts at node index 0, we must offset the consecutive ones.
    current_edge_index = 0

    for _, labeled_edges in edges_iterable:
        # Offset the node ids in relation to the previous subgraphs
        edges = labeled_edges[:, [0, 2]].astype(int)  # Remove the labels
        edge_labels = labeled_edges[:, 1]

        # Node index offsetting
        max_index = edges.max() + 1
        edges += node_count
        node_count += max_index

        # Fill the rows
        for edge, label in zip(edges, edge_labels):
            label = label_lookup[label]

            # Forward edge
            rows[current_edge_index] = edge[0]
            cols[current_edge_index] = edge[1]
            depths[current_edge_index] = label

            # Backward edge
            rows[current_edge_index + 1] = edge[1]
            cols[current_edge_index + 1] = edge[0]
            depths[current_edge_index + 1] = label + 16

    # Create the sparse tensor holding the adjacency matrix
    # TODO : matrix preprocessing operations ?
    tensor_shape = (node_count, node_count, 32)

    sparse_3d_tensor = torch.sparse_coo_tensor(
        indices=torch.stack([depths, rows, cols]),
        values=data,  # type: ignore
        size=tensor_shape,
    )

    return sparse_3d_tensor


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
    with torch.no_grad():
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
    model.train()
