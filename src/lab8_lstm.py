import torch
import torch.nn.functional as F
from torch import nn


class Lab8_LSTM(nn.Module):
    """Embedding LSTM model based on the 2nd model of Lab8.
    This model uses text data only (no graph data).
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_class: int):
        """Instanciate a Lab8_LSTM Model.

        Params:
            - vocab_size (int) : the number of words in the text embedder vocabulary.
            - embed_dim (int) : the dimension of the model's internal embedding space.
            - hidden_dim (int) : the number of neurons for the hidden layer inside the LSTM cell.
            - num_class (int) : the number of output classes. For binary classification, this parameter must be set to 1.
        """
        super().__init__()
        # Layers
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, text: torch.Tensor):
        output, _ = self.lstm(text)

        x = self.fc(output)  # Linear layer

        return (
            F.sigmoid(x),
            output,
        )  # Binary classification : sigmoid [0,1]. Also return the hidden state embeddings


if __name__ == "__main__":
    # Test dictionary generation
    from loader import get_device, get_train_test_split_sets
    from sentence_transformers import SentenceTransformer
    from utils import train_model
    from word_embedding import DictionaryEmbedder, dictionary_from_data

    # Load the data
    (
        X_train,
        y_train,
        train_adjacency_matrix,
        X_test,
        y_test,
        test_adjacency_matrix,
    ) = get_train_test_split_sets(0.2)

    # Prepare both text embedders
    words = dictionary_from_data(X_train, y_train, percentile=95, score_threshold=0.5)
    embedder = DictionaryEmbedder(words)
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode all data
    X_train_bert = bert.encode(X_train, show_progress_bar=True)
    X_test_bert = bert.encode(X_test, show_progress_bar=True)
    X_train_dict = embedder.encode_batch(X_train, show_progress=True)
    X_test_dict = embedder.encode_batch(X_test, show_progress=True)

    # Instanciate nn models
    hidden_dim = 32  # TODO : update this parameter, and the hidden layers of the LSTM
    bert_lstm = Lab8_LSTM(X_train_bert[0].shape[0], hidden_dim, 1)
    dict_lstm = Lab8_LSTM(embedder.size(), hidden_dim, 1)

    # Move to GPU
    device = get_device()
    # Move models
    bert_lstm.to(device)
    dict_lstm.to(device)
    # Move X data
    X_train_bert = torch.FloatTensor(X_train_bert).to(device)
    X_train_dict = torch.FloatTensor(X_train_dict).to(device)
    X_test_bert = torch.FloatTensor(X_test_bert).to(device)
    X_test_dict = torch.FloatTensor(X_test_dict).to(device)
    # Move y labels
    y_train = torch.FloatTensor(y_train).to(device).view(-1, 1)
    y_test = torch.FloatTensor(y_test).to(device).view(-1, 1)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    learning_rate = 0.01
    bert_optimizer = torch.optim.Adam(bert_lstm.parameters(), lr=learning_rate)
    dict_optimizer = torch.optim.Adam(dict_lstm.parameters(), lr=learning_rate)

    # Train the models
    print("Training BERT LSTM model...")
    train_model(
        bert_lstm,
        bert_optimizer,
        criterion,
        X_train_bert,
        y_train,
        epochs=100,
    )
    print("Training Dictionary LSTM model...")
    train_model(
        dict_lstm,
        dict_optimizer,
        criterion,
        X_train_dict,
        y_train,
        epochs=100,
    )
