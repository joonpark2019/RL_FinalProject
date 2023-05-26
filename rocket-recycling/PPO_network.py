import random
import numpy as np
import torch
import utils
import torch.optim as optim

import torch.nn as nn

class VNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcV1 = torch.nn.Linear(8, 256)
        self.fcV2 = torch.nn.Linear(256, 256)
        self.fcV3 = torch.nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.fcV1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcV2(x)
        x = torch.nn.functional.relu(x)
        x = self.fcV3(x)
        return x
    
class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcA1 = torch.nn.Linear(8, 256)
        self.fcA2 = torch.nn.Linear(256, 256)
        self.fcA3_thrust = torch.nn.Linear(256, 3)
        self.fcA3 = torch.nn.Linear(256, 9)
        
    def forward(self, x):
        x = self.fcA1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcA2(x)
        x = torch.nn.functional.relu(x)
        x = self.fcA3(x)  
        x = torch.nn.functional.softmax(x, dim=-1)
        return x
    