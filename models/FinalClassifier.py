from torch import nn
from utils.logger import logger

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}
    

class LSTM(nn.Module):
    def __init__(self, num_classes):
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
        super().__init__()
        self.input_size = 1024
        self.hidden_size = 128
        self.num_layers = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, num_classes, batch_first=True)

    def forward(self, x):
        logger.info(f"x parameter: {x}")
        return self.lstm(x), {}
    


