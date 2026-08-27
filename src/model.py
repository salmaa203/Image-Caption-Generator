# import torch
import torch.nn as nn


class ImageCaptioningModel(nn.Module):

    def __init__(
        self,
        feature_dim,
        vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=1,
        dropout=0.3,
        pad_idx=0
    ):
        super().__init__()

        # Project image features: 2048 -> 256
        self.feature_projection = nn.Linear(
            feature_dim,
            embed_dim
        )

        # Word embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout
        )

        # Vocabulary prediction
        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize hidden state from image features
        self.init_h = nn.Linear(
            feature_dim,
            hidden_dim
        )

        # Initialize cell state from image features
        self.init_c = nn.Linear(
            feature_dim,
            hidden_dim
        )

    def forward(self, features, captions):

        # Initialize hidden and cell states
        h = self.init_h(features)
        c = self.init_c(features)

        h = h.unsqueeze(0)
        c = c.unsqueeze(0)

        # Remove <end> token from input
        inputs = captions[:, :-1]

        # Word embeddings
        embeddings = self.embedding(inputs)

        embeddings = self.dropout(embeddings)

        # LSTM
        outputs, _ = self.lstm(
            embeddings,
            (h, c)
        )

        # Predict vocabulary distribution
        outputs = self.fc(outputs)

        return outputs