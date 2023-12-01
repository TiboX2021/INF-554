from typing import Iterable

import numpy as np
import scipy.sparse as sp
import torch

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
        total_edges = np.concatenate((total_edges, edges))

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
