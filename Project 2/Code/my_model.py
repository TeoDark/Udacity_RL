import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Actor, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)
        self.elu = nn.ELU()
        #self.tanh = nn.Tanh()
        #self.fc4 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        mu = self.fc3(x)
        #x = self.tanh(self.fc3(x))

        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd

class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.elu = nn.ELU()
        #self.tanh = nn.Tanh()
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        #x = self.tanh(self.fc1(x))
        #x = self.tanh(self.fc2(x))
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        v = self.fc3(x)
        return v
    