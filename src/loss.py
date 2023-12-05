"""
Custom loss function that maximizes f1 score
"""
import torch
import torch.nn as nn
from torch import Tensor


class F1Loss(nn.Module):
    # TODO : INIT

    def forward(self, output: Tensor, target: Tensor):
        # TODO : vérifier que le calcul est correct
        target = torch.round(target)
        tp = torch.sum(output * target, dim=0)
        tn = torch.sum((1 - output) * (1 - target), dim=0)
        fp = torch.sum((1 - output) * target, dim=0)
        fn = torch.sum(output * (1 - target), dim=0)

        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)

        f1 = 2 * p * r / (p + r + 1e-7)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return 1 - torch.mean(f1)


if __name__ == "__main__":
    from lab8_lstm import Lab8_LSTM
    from loader import get_device, get_full_training_sets
    from sentence_transformers import SentenceTransformer
    from sklearn.model_selection import train_test_split
    from utils import test_model, train_model

    # Text-only model : we do not need the adjacency matrix
    X_full, y_full = get_full_training_sets()

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    # Embed utterances using bert embedder
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = bert.encode(X_train, show_progress_bar=True)
    X_test = bert.encode(X_test, show_progress_bar=True)

    # Instanciate models
    # TODO : test with optimized RNNs
    embed_dim = X_train[0].shape[0]
    hidden_dim = 64
    bce_model = Lab8_LSTM(embed_dim, hidden_dim, 1)
    f1_model = Lab8_LSTM(embed_dim, hidden_dim, 1)

    # Move everything to GPU
    device = get_device()
    bce_model.to(device)
    f1_model.to(device)
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.FloatTensor(y_train).to(device).view(-1, 1)
    y_test = torch.FloatTensor(y_test).to(device).view(-1, 1)

    # Instanciate optimizers and loss functions
    learning_rate = 0.001
    bce_optimizer = torch.optim.Adam(bce_model.parameters(), lr=learning_rate)
    f1_optimizer = torch.optim.Adam(f1_model.parameters(), lr=learning_rate)

    bce_loss = nn.BCELoss()
    f1_loss = F1Loss()

    epochs = 100

    # Train both models
    train_model(
        bce_model,
        bce_optimizer,
        bce_loss,
        X_train,
        y_train,
        epochs=epochs,
    )

    train_model(
        f1_model,
        f1_optimizer,
        f1_loss,
        X_train,
        y_train,
        epochs=epochs,
    )

    # Test both models
    print("Testing BCE model")
    test_model(bce_model, X_test, y_test)
    print("Testing F1 model")
    test_model(f1_model, X_test, y_test)
