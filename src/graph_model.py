"""
Graph-based model

TODO : use heterogeneous graph data structure
"""


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
    train_model(model, optimizer, criterion, X_train, y_train, 130, X_test, y_test)

    # Test model
    test_model(model, X_train, y_train, False)
    test_model(model, X_test, y_test, False)
