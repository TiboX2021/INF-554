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


class Lab8_LSTM(nn.Module):
    """Embedding LSTM model based on the 2nd model of Lab8.
    This model uses text data only (no graph data).
    """

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, num_class: int
    ):
        """Instanciate a Lab8_LSTM Model.

        Params:
            - vocab_size (int) : the number of words in the text embedder vocabulary.
            - embed_dim (int) : the dimension of the model's internal embedding space.
            - hidden_dim (int) : the number of neurons for the hidden layer inside the LSTM cell.
            - num_class (int) : the number of output classes. For binary classification, this parameter must be set to 1.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, text: torch.Tensor):
        # 1st embedding layer
        text = self.embedding(text)

        _output, (hidden, _cell) = self.lstm(text)

        x = hidden.view(-1, self.hidden_dim)

        x = self.fc(x)  # Linear layer

        return (
            F.sigmoid(x),
            hidden,
        )  # Binary classification : sigmoid [0,1]. Also return the hidden state embeddings


if __name__ == "__main__":
    pass
    # TODO : test dropout and weight decay against overfitting. Use sklearn metrics (f1, accuracy, confusion matrix) to evaluate the models
