import torch.nn as nn

class Model(nn.Module):
  def __init__(self, 
               input_size = 1, 
               hidden_size = 30, 
               num_layers = 1, 
               ):
    super(Model, self).__init__()
    self.lstm = nn.LSTM(input_size = input_size, 
                        hidden_size = hidden_size,
                        num_layers = num_layers,
                        batch_first = True)
    self.fc = nn.Linear(hidden_size, 1)


  def forward(self, batch):
    batch = batch.to(next(self.parameters()).device)
    _, (hn, cn) = self.lstm(batch)
    output = self.fc(hn[-1].squeeze()).squeeze()
    return output
