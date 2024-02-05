import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.logger import logger


class TransformerClassifier(nn.Module):
    def __init__(
        self, num_classes, input_size=1024, num_heads=8, hidden_size=1024, num_layers=6
    ):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layers = TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # * Embedding: Linear transformation of input features
        x = self.embedding(x)

        # * Reshape input tensor for Transformer encoder
        # addind a dimension -> from (batch_size, hidden_size) to (batch_size, 1, hidden_size)
        x = x.unsqueeze(1)
        # resulting shape (1, batch_size, hidden_size) -> where 1 is the sequence length
        x = x.permute(1, 0, 2)

        # * Forward pass through transformer encoder
        transformer_output = self.transformer_encoder(x)

        # * Extract features and compute logits -> Taking output of the first token as features
        feat = transformer_output[0, :, :]

        # * Compute logits
        logits = self.fc(feat)  # Fully connected layer for logits

        return logits, {"features": feat}
