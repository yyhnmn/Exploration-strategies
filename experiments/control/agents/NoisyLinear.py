# import torch
# import torch.nn as nn
# import torch.nn.functional as f
# from torch.autograd import Variable
# import math

# https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb

# class NoisyLinear(nn.Module):
#     def __init__(self, input_size,output_size):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.std_init = 0.4
#         self.weight_mu    = nn.Parameter(torch.FloatTensor(output_size, input_size))  
#         self.weight_sigma = nn.Parameter(torch.FloatTensor(output_size, input_size))
#         self.register_buffer('weight_epsilon', torch.FloatTensor(output_size, input_size))
        
#         self.bias_mu    = nn.Parameter(torch.FloatTensor(output_size))
#         self.bias_sigma = nn.Parameter(torch.FloatTensor(output_size))
#         self.register_buffer('bias_epsilon', torch.FloatTensor(output_size))
        
#         self.reset_parameters()
#         self.reset_noise()

#     def forward(self, x):
#         if self.training: 
#             weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
#             bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
#         else:
#             weight = self.weight_mu
#             bias   = self.bias_mu
        
#         return f.linear(x, weight, bias)

#     def reset_parameters(self):
#         mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
#         self.weight_mu.data.uniform_(-mu_range, mu_range)
#         self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
#         self.bias_mu.data.uniform_(-mu_range, mu_range)
#         self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
#     def reset_noise(self):
#         epsilon_in  = self._scale_noise(self.input_size)
#         epsilon_out = self._scale_noise(self.output_size)
        
#         self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
#         self.bias_epsilon.copy_(self._scale_noise(self.output_size))
    
#     def _scale_noise(self, size):
#         x = torch.randn(size)
#         x = x.sign().mul(x.abs().sqrt())
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import math

# https://github.com/BY571/DQN-Atari-Agents/blob/master/Agents/Networks/DQN.py

class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # extra parameter for the bias and register buffer for the bias parameter
        if bias: 
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
    
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    
    def forward(self, input):
        # sample random noise in sigma weight buffer and bias buffer
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return f.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)