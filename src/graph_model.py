"""
Graph-based model
TODO : currently, only forward edges are implemented. We can add 16 more dimensions for backward edges.
"""
from typing import TypedDict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv


class GraphOutput(TypedDict):
    """Output of the graph's forward pass"""

    utterances: Tensor


class GNN(nn.Module):
    """
    TODO : try different graph models. This one was taken from the torch_geometric heterogeneous graphs example.
    This is a homogeneous graph model, but it can be converted to a heterogeneous graph model using the to_hetero function.
    """

    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        # SAGE Conv samples information from neighbor nodes. It is a graph convolutional layer.
        # NOTE : we may need only one ? A voir...
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x: Tensor, edge_index: Tensor):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return F.sigmoid(x)  # Sigmoid for binary classification


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from loader import geometric_train_test_split, get_device
    from rnn import RNNClassifier
    from utils import test_model, train_model

    (
        X_train,
        y_train,
        edges_train,
        X_test,
        y_test,
        edges_test,
    ) = geometric_train_test_split("all-mpnet-base-v2")

    device = get_device()

    print(X_train.shape)
    print(y_train.shape)
    print(edges_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(edges_test.shape)

    ######################################################################################################
    #                                 BERT EMBEDDING TO RNN EMBEDDING                                    #
    ######################################################################################################
    """In this part, we use BERT embeddings and produce embeddings in dimension 32 using a RNN.
    These embeddings can produce a f1_score of up to 0.56 using an additional linear layer + sigmoid
    """

    # Instanciate model
    input_size = X_train.shape[1]  # Embeddings size
    hidden_size = 32
    num_layers = 2
    model = RNNClassifier(input_size, hidden_size, num_layers)

    # Chose optimizer and loss function
    criterion = nn.BCELoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move everything to GPU
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Train model
    # DEBUG : remettre les epochs à 150. Pour l'instant à 10 pour debug le training du graphe
    train_model(model, optimizer, criterion, X_train, y_train, 10, X_test, y_test)

    # Test model
    test_model(model, X_train, y_train, False)
    test_model(model, X_test, y_test, False)

    ######################################################################################################
    #                                       RNN EMBEDDING TO GRAPH                                       #
    ######################################################################################################
    """In this part, we use the produced RNN embeddings to feed a graph-based model.
    """
    from torch_geometric.nn import to_hetero
    from utils import build_hetero_data, test_graph_model

    # Compute embeddings from the RNN model
    model.eval()
    _, X_train = model.forward(X_train)
    _, X_test = model.forward(X_test)

    # Detach the tensors from their previous model
    y_train = y_train.detach()
    y_test = y_test.detach()
    X_train = X_train.detach()
    X_test = X_test.detach()

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Produce the heterogenous data for the heterogeneous graph model
    train_hetero_data = build_hetero_data(X_train, edges_train)
    test_hetero_data = build_hetero_data(X_test, edges_test)

    # Transfer data to GPU
    train_hetero_data = train_hetero_data.to(device)  # type: ignore
    test_hetero_data = test_hetero_data.to(device)  # type: ignore

    graph_model = GNN(hidden_channels=hidden_size, out_channels=1)
    graph_model: GNN = to_hetero(graph_model, train_hetero_data.metadata(), aggr="sum")

    # Chose optimizer and loss function
    criterion = nn.BCELoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(graph_model.parameters(), lr=learning_rate)

    # Move everything to GPU
    graph_model.to(device)

    ######################################################################################################
    #                                          GRAPH  TRAINING                                           #
    ######################################################################################################

    graph_epochs = 150

    for epoch in range(graph_epochs):
        graph_model.train()
        optimizer.zero_grad()
        out: GraphOutput = graph_model(
            train_hetero_data.x_dict, train_hetero_data.edge_index_dict
        )

        loss = criterion(out["utterances"], y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} : {loss.item()}", end=" --- ")
            test_graph_model(graph_model, test_hetero_data, y_test)
