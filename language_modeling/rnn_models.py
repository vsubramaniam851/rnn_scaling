import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNLM(nn.Module):
    def __init__(self, rnn_model, vocab_size, embedding_dim, hidden_dim, num_layers, device):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if rnn_model == 'LSTM':
            self.rnn = nn.ModuleList([
                nn.LSTM(embedding_dim if i == 0 else hidden_dim, hidden_dim, num_layers = 1, batch_first = True)
                for i in range(num_layers)
            ])
        else:
            self.rnn = nn.ModuleList([
                nn.RNN(embedding_dim if i == 0 else hidden_dim, hidden_dim, num_layers = 1, batch_first = True, nonlinearity='tanh')
                for i in range(num_layers)
            ])
        self.fc1 = nn.Linear(hidden_dim, vocab_size)

        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn_model = rnn_model

    def forward(self, x, hidden = None, **kwargs):
        if hidden == None:
            hidden = self.init_hidden(x.shape[0])
        x = self.embedding(x)
        if self.rnn_model == 'LSTM':
            for _, lstm_layer in enumerate(self.rnn):
                x, hidden = lstm_layer(x, hidden)
        else:
            for _, rnn_layer in enumerate(self.rnn):
                x, hidden = rnn_layer(x, hidden[0].unsqueeze(0) if len(hidden[0].shape) == 2 else hidden[0])
                hidden = tuple(hidden, )
        out = self.fc1(x)
        return out, hidden

    def init_hidden(self, batch_size):
            return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                    torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

if __name__ == '__main__':
    pass