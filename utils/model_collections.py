import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_REG(nn.Module):
    def __init__(self, feature_dim, hidden_dim, reg_dim, device ,dropout=0,
                    num_layers=2, bidirectional=True, batch_size=1):
        super(RNN_REG, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_directions = 2 if bidirectional == True else 1
        self.dropout = dropout
        self.device = device
        
        self.rnn = nn.RNN(self.feature_dim, self.hidden_dim, num_layers=num_layers,
                         dropout=dropout, bidirectional=bidirectional).double()
        
        self.hidden2reg = nn.Linear(hidden_dim  * self.num_directions, reg_dim).double()
        self.hidden = self.init_hidden()
        
        
    def init_hidden(self):
        return torch.zeros(self.num_layers * self.num_directions, 
                           self.batch_size, self.hidden_dim, device=self.device).double()
                
    
    def forward(self, inputs):
        #inputs = torch.from_numpy(inputs).float()
        rnn_out, self.hidden = self.rnn(inputs, self.hidden)
        if self.dropout > 0:
            rnn_out = nn.Dropout(p=self.dropout)(rnn_out)
        reg = self.hidden2reg(rnn_out)
        return reg 

class FC_REG(nn.Module):
    def __init__(self, feature_dim, hidden_dim, reg_dim, batch_size=1):
        super(FC_REG, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.batch_size = batch_size

        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.hidden2reg = nn.Linear(hidden_dim * 2, reg_dim)


    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        reg = self.hidden2reg(x)
        return reg

class FC_Relu_REG(nn.Module):
    def __init__(self, feature_dim, hidden_dim, reg_dim, batch_size=1):
        super(FC_Relu_REG, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.batch_size = batch_size

        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.hidden2reg = nn.Linear(hidden_dim * 2, reg_dim)
        self.relu = nn.ReLU()


    def forward(self, inputs):
        #inputs = torch.from_numpy(inputs).float()
        x = self.linear1(inputs)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        reg = self.hidden2reg(x)
        return reg
