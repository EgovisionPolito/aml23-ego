import torch
from torch import nn
from utils.logger import logger

class LSTM(nn.Module):
    def __init__(self, num_classes=8): #* aggiusta i parametri, ad es. passa la batch come arg
        super(LSTM, self).__init__()
        self.input_size = 1024
        self.hidden_size = 32
        self.num_layers = 1
        #self.sequence_length = 1 # quanti x gli passo, credo 1024 (cioè le colonne)
        self.batch_size = 32 # da prendere nello yaml
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states with the proper batch size
        # h0 and c0 shape = (num_layers, batch_size, hidden_size)=(1, 32, 32)
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(x.device)

        # aggiungo una dimensione 
        # we want x shape equal to (batch_size, sequence_length, input_size)=(32, 1, 1024)
        # prima: x.shape = (32, 1024)
        x = x.unsqueeze(1)
        # dopo: x.shape = (32, 1, 1024)

        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out: contains the output features (batch_size, sequence_length, hidden_size)=(32, 1, 32)
        # hn: final hidden state for each element in sequence, stessa size di h0
        # cn: final cell state for each element in sequence, stessa di c0

        # Reshape the output to be compatible with the fully connected layer
        feat = out.view(-1, self.hidden_size)
        # Pass through fully connected layer to get logits
        logits = self.fc(feat)
        logger.info(f"######## => x.size(0): {x.size(0)} | bs: {self.batch_size} | x: {x} | x.shape: {x.shape} | l.shape: {logits.shape} | f.shape: {feat.shape} | logits: {logits} | feat: {feat}")
        
        return logits, {"features": feat} 
    
    
    #(32, 8), (32, 1024)
    # last_output = out[:, -1, :]
    # logits = self.fc(last_output)
    # last_hn = out[:, -1, :]
    # feat = self.fc(last_hn)
    # #feat = hn[-1] #(32, 32)
    # #logits = self.fc(out) #(32, 8)
    # logger.info(f"hn: {hn}, hn[-1]=feat: {feat}, shape: {feat.shape}, logits: {logits}, shape: {logits.shape}")  #logits: tipo le label, cioè le previsioni tipo


