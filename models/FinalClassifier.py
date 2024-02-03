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
    def __init__(self, num_classes=8):
        super(LSTM, self).__init__()
        self.input_size = 1024
        self.hidden_size = 128
        self.num_layers = 1
        self.sequence_length = 1024
        self.batch_size = 32 # da prendere nello yaml
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        logger.info(f"x parameter: {x}")
        reshaped_x = x.reshape(self.batch_size, self.sequence_length, -1)
        logger.info(f"x parameter: {x}, length: {len(x)}, reshaped_x: {reshaped_x}")
        # lstm_out, _ = self.lstm(x)
        # output = self.fc(lstm_out[:, -1, :])  # Assuming you want to use the output from the last time step
        # features = {'lstm_out': lstm_out}  # Modify this to include any other intermediate features
        # return output, features
        return self.lstm(reshaped_x), {}
    
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

