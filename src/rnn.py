"""
Fine-tuned big RNN model for best performance on text-only data.

NOTE :
2 LSTM layers with 32 hidden dimensions do not go further than 0.54 f1_score on the testing set.
It is not necessary to try bigger models or bigger embedders, this score cannot be outdone (sometimes 0.55 pops up, but not better).

The last optimizations we can do are exclusively based on the graph adjacency.
"""
import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.5,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        hidden, _ = self.rnn(x)
        hidden = self.layer_norm(hidden)
        out = self.fc(hidden)
        out = self.sigmoid(out)
        return out, hidden


if __name__ == "__main__":
    from loader import get_device, get_full_preembedded_training_sets
    from sklearn.model_selection import train_test_split
    from utils import test_model, train_model

    # Load data
    device = get_device()

    # We load preembedded values from the large BERT embedder
    X_full, y_full = get_full_preembedded_training_sets("all-MiniLM-L6-v2")

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

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
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.FloatTensor(y_train).to(device).view(-1, 1)
    y_test = torch.FloatTensor(y_test).to(device).view(-1, 1)

    # Train model - pass X_test and y_test to monitor f1_score over the testing set
    train_model(model, optimizer, criterion, X_train, y_train, 200, X_test, y_test)

    # Test model
    test_model(model, X_train, y_train)
    test_model(model, X_test, y_test)

    # Save model
    torch.save(model.state_dict(), "models/rnn_model.pt")
