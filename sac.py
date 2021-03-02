
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import time
from collections import OrderedDict



class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,log_std_min=-2, log_std_max=2,num_action=1):
        """Initialize parameters and build model.
        Params
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max



        
        # predict angle [-1,1] for each-pixel : OUT = 20x20
        self.actornet = nn.Sequential(OrderedDict([
            ('actor-norm0', nn.BatchNorm2d(1024)),
            ('actor-relu0', nn.ReLU(inplace=True)),
            ('actor-conv0', nn.Conv2d(1024, 128, kernel_size=3, stride=1,padding=1, bias=False)),
            ('actor-norm1', nn.BatchNorm2d(128)),
            ('actor-relu1', nn.ReLU(inplace=True)),
            ('actor-conv1', nn.Conv2d(128, 64, kernel_size=3, stride=1,padding=1,bias=False)),
            ('actor-norm2', nn.BatchNorm2d(64)),
            ('actor-relu2', nn.ReLU(inplace=True)),
            ('actor-conv2', nn.Conv2d(64, 32, kernel_size=1, stride=1,bias=False)),
            ('actor-norm3', nn.BatchNorm2d(32)),
            ('actor-relu3', nn.ReLU(inplace=True))

        ]))

        self.mu = nn.Conv2d(32, num_action, kernel_size=1, stride=1, bias=False)
        self.log_std_linear = nn.Conv2d(32, num_action, kernel_size=1, stride=1, bias=False)

    def forward(self, state):
        x = self.actornet(state)

        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        dist = Normal(0, 1)
        e = dist.sample().cuda()
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob , torch.tanh(mu)
        
    
    def get_angle(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        #state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        dist = Normal(0, 1)
        e      = dist.sample().cuda()
        action = torch.tanh(mu + e * std).cpu()


        return action,torch.tanh(mu)


class Critic(nn.Module):

    def __init__(self,num_action=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()


        self.critic = nn.Sequential(OrderedDict([
            ('critic-norm0', nn.BatchNorm2d(1024 + num_action)),
            ('critic-relu0', nn.ReLU(inplace=True)),
            ('critic-conv0', nn.Conv2d(1024 + num_action, 128, kernel_size=1, stride=1, bias=False)),
            ('critic-norm1', nn.BatchNorm2d(128)),
            ('critic-relu1', nn.ReLU(inplace=True)),
            ('critic-conv1', nn.Conv2d(128, 64 , kernel_size=1, stride=1,bias=False)),
            ('critic-norm2', nn.BatchNorm2d(64)),
            ('critic-relu2', nn.ReLU(inplace=True)),
            ('critic-conv2', nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=False)),
            ('critic-norm3', nn.BatchNorm2d(32)),
            ('critic-relu3', nn.ReLU(inplace=True)),
            ('critic-conv3', nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False))

        ]))



    

    def forward(self, states,actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x  = torch.cat((states, actions.float()), dim=1)


        x = self.critic(x)
        return x


