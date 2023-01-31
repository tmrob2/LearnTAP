import torch.nn as nn
import torch.nn.functional as F
import torch
"""
We can build some fancier models here if we need to build some sort of
recurrence for partial observability but a DQN should be fine for deep 
see treasure (DST). If we don't assume partial observability then DQN is fine. 

"""

# A fully connected network. The input shape should be
# [obj, state] which is just repeating the state O times
# for each objective
class QObj(nn.Module):
    def __init__(self, inputs, hidden, actions):
        super(QObj, self).__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Returns the \bar{Q}(s)[a1,...,ak]
        # i,.e. the shape will be [obj, num_actions]
        # the loss will then be the Huber distance between
        # \bar{Q}, \bar{Q}'
        return x

        
