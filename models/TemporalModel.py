import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class TemporalModel(nn.Module):
    def __init__(
        self,
        input_channels,
        conv1d_channels,
        mode="avg",
    ):
        super(TemporalModel, self).__init__()

        self.mode = mode

        # Activation function
        self.relu = nn.ReLU()

        if self.mode == "conv":
            # Convolutional layer
            self.aggregate = nn.Conv1d(
                in_channels=input_channels,
                out_channels=conv1d_channels,
                kernel_size=5,
                stride=1,
                bias=False,
            )
            # Initialize weights
            init.xavier_uniform_(self.aggregate.weight)

    def forward(self, x):

        if self.mode == "avg":
            # Apply AvgPooling
            x = F.avg_pool1d(x.unsqueeze(0).transpose(1, 2), kernel_size=5).squeeze(2)

        elif self.mode == "conv":
            # Apply Conv1d
            x = self.aggregate(x)
            # Apply Activation
            x = self.relu(x)

        return x
