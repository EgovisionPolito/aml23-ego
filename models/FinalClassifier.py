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
      self.input_size = 1024
      self.hidden_size = 32
      self.num_layers = 1
      #self.sequence_length = 1 # quanti x gli passo, credo 1024 (cioè le colonne)
      self.batch_size = 32 # da prendere nello yaml
      self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
      # self.fc = nn.Linear(self.hidden_size, num_classes)

  def forward(self, x):
      #* we want x shape equal to (batch_size, sequence_length, input_size)
      logger.info(f"x_shape: {x}")
      x = x.unsqueeze(1)
      #x.reshape(self.batch_size, 1, self.input_size)
      # Initialize hidden and cell states with the proper batch size
      h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(x.device)
      c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(x.device)

      # Forward pass through LSTM
      out, (hn, cn) = self.lstm(x, (h0, c0))
      logger.info(f"out_shape: {out}")  #logits: tipo le label, cioè le previsioni tipo

      #self.fc(out) -> forse per le logits

      return logits, {"features": feat} 