import torch
from torch import nn
from utils.logger import logger

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}
    

class LSTM(nn.Module):
    def __init__(self, num_classes=8):
        super(LSTM, self).__init__()
        # self.input_size = 32
        self.input_size = 1024
        self.hidden_size = 32
        self.num_layers = 1
        self.sequence_length = 1 # quanti x gli passo, credo 1024 (cioè le colonne)
        # self.sequence_length = 1024 # quanti x gli passo, credo 1024 (cioè le colonne)
        self.batch_size = 32 # da prendere nello yaml
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        #* we want x shape equal to (batch_size, sequence_length, input_size)
        # reshaped_x = x.reshape((1, 1, 32))
        # # x.reshape((self.batch_size, self.sequence_length, self.input_size))
        # logger.info(f"x: {reshaped_x}, x_shape: {reshaped_x.shape}")
        # h0 = torch.zeros(self.num_layers, reshaped_x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, reshaped_x.size(0), self.hidden_size).to(device)
        # out, _ = self.lstm(reshaped_x, (h0, c0))
        # logger.info(f"OUT: {out}, OUT_SHAPE: {out.shape}")
                # Check the shape of x
        logger.info(f"x_shape: {x.shape}")

        # Assuming x has shape (32), reshape it for LSTM
        reshaped_x = x.view(self.batch_size, 1, self.input_size)
        logger.info(f"reshaped_x_shape: {reshaped_x.shape}")

        # Initialize hidden and cell states with the proper batch size
        h0 = torch.zeros(self.num_layers, reshaped_x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, reshaped_x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        out, hn = self.lstm(reshaped_x, (h0, c0))
        logger.info(f"out_shape: {out.shape}")

        return out, hn
    
        '''
        Parameters
            input_size - The number of expected features in the input x

            hidden_size - The number of features in the hidden state h

            num_layers - Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1

            bias - If False, then the layer does not use bias weights b_ih and b_hh. Default: True

            batch_first - If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False

            dropout - If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0

            bidirectional - If True, becomes a bidirectional LSTM. Default: False

            proj_size - If > 0, will use LSTM with projections of corresponding size. Default: 0
        '''

