import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalModel(nn.Module):
    def __init__(
        self,
        input_channels,
        output_classes,
        conv1d_channels,
        fc_hidden_units,
        sequence_length,
        use_maxpool=True,
    ):
        super(TemporalModel, self).__init__()

        # Convolutional layer
        self.conv1d = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv1d_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Activation function
        self.relu = nn.ReLU()

        # MaxPooling layer (optional)
        self.use_maxpool = use_maxpool
        if self.use_maxpool:
            self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
            sequence_length //= 2

        # Fully connected layers
        self.fc1 = nn.Linear(conv1d_channels * sequence_length, fc_hidden_units)
        self.fc2 = nn.Linear(fc_hidden_units, output_classes)

    def forward(self, x):
        # Input x should have shape (batch_size, input_channels, sequence_length)

        # Apply Conv1d
        x = self.conv1d(x)

        # Apply Activation
        x = self.relu(x)

        # Apply MaxPooling (optional)
        if self.use_maxpool:
            x = self.maxpool(x)

        # Flatten the output before fully connected layers
        x = x.view(x.size(0), -1)

        # Apply Fully Connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
