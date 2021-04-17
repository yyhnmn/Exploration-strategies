import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import math

# https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb

class NoisyLinear(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.std_init = 0.4
        self.weight_mu    = nn.Parameter(torch.FloatTensor(output_size, input_size))  
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_size, input_size))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_size, input_size))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(output_size))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_size))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_size))
        
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return f.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.input_size)
        epsilon_out = self._scale_noise(self.output_size)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_size))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x