import torch as torch 
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    '''a helper function to initialize the weights of the model'''
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    """Actor model for the DDPG algorithm.
    
    Layer 1: 400 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in)
    Layer 2: 300 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in)
    Layer 3: 1 unit, tanh activation, intialized with uniform weights in the range of -0.003 to 0.003
    
    """
    def __init__(self, input_size:tuple[int], action_size:int,CNN = None):
        """
        input: tuple[int]
            The input size, as a tuple of dimensions, for the DoubleInvertedPendulum environment, of shape (11,)
        action_size: int
            The number of actions
        """
        super(Actor, self).__init__()
        # ========== YOUR CODE HERE ==========
        # TODO:
        # define the fully connected layers for the actor
        # ====================================
        self.input_size = input_size
        self.action_size = action_size
        self.CNN = CNN
        self.fc1 = nn.Linear(input_size[0], 400)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(400, 300)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(300, 1)
        self.tanh = nn.Tanh()
        self.init_weights()
    
        # ========== YOUR CODE ENDS ==========
        
    def init_weights(self,init_w=3e-3):
        """
        Args:
            init_w (float, optional): the onesided range of the uniform distribution for the final layer. Defaults to 3e-3.
        """
        # ========== YOUR CODE HERE ==========
        # TODO:
        # initialize the weights of the model
        # ====================================
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size(), self.input_size[0])
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size(), 400)
        self.fc3.weight.data.uniform_(-init_w, init_w)    
        # ========== YOUR CODE ENDS ==========
    
        # ========== YOUR CODE ENDS ==========
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # ========== YOUR CODE HERE ==========
        x = x.view(-1, self.input_size[0])
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x
        # ========== YOUR CODE ENDS ==========
    


class Critic(nn.Module):
    """Critic model for the DDPG algorithm.
    Layer 1: 400 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in)
    Layer 2: 300 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in). Input is the concatenation of the 400 dimension embedding from the state, and the action taken.
    Layer 3: 1 unit, intialized with uniform weights in the range of -0.003 to 0.003
    """
    def __init__(self,input_size:tuple[int],action_size:int):
        """
        input: tuple[int]
            The input size, as a tuple of dimensions, for the DoubleInvertedPendulum environment, of shape (11,)
        action_size: int
            The number of actions
        """
        super(Critic, self).__init__()
        # ========== YOUR CODE HERE ==========
        # TODO: 
        # define the fully connected layers for the critic and initialize the weights
        # ====================================
        self.fc1 = nn.Linear(input_size[0], 400)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(400 + action_size, 300)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(300, 1)
        self.input_size = input_size
        self.action_size = action_size
        self.init_weights()
    
        # ========== YOUR CODE ENDS ==========
        
    def init_weights(self,init_w=3e-3):
        # ========== YOUR CODE HERE ==========
        # TODO:
        # initialize the weights of the model
        # ====================================
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size(), self.input_size[0])
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size(), 400 + self.action_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        

    
        # ========== YOUR CODE ENDS ==========
    
        # ========== YOUR CODE ENDS ==========
        
    def forward(self, x:torch.Tensor, a:torch.Tensor)->torch.Tensor:
        # ========== YOUR CODE HERE ==========
        x = x.view(-1, self.input_size[0])
        x = self.fc1(x)
        x = self.relu1(x)
        a = a.view(-1, self.action_size)
        x = torch.cat((x, a), dim=1)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
            
    
        # ========== YOUR CODE ENDS ==========