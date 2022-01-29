import torch.nn as nn
import torch
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net(nn.Module):
    def __init__(self, vocab_size, bidirectional = True, num_classes = 1, embedding_dim = 128, lstm_out_dim = 128):
        print("Building NN...")
        
        super().__init__()
        self.l1 = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.l2 = nn.LSTM(embedding_dim, lstm_out_dim, dropout = 0.2, num_layers = 2, bidirectional = bidirectional)
        self.l3 = nn.Flatten()
        self.l4 = nn.Linear(128 * lstm_out_dim * 2, num_classes)
    
    def forward(self, x):
        #Aqui eu só to passando o input pelas camadas mesmo
        x    = self.l1(x)
        x, _ = self.l2(x)
        x    = self.l3(x)
        x    = self.l4(x)
#         x    = F.softmax(x, dim = -1)
            
        return x