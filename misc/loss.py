import pdb
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

def rl_loss(q_values, log_probs, baselines, critic_enabled=False, critic_coeff=1e-2, entropies=None, entropy_coeff=1e-4, debug=False):
    """
    Computes REINFORCE / Actor-Critic losses based on the inputs provided
    INPUTS:
       q_values       - q values estimated for each step of the trajectory using observed samples
                      - BxL pytorch Tensor
       log_probs      - log probability of action executed by the policy at each time step
                      - BxL pytorch Variable
       baselines      - estimated baseline at each step; it is the average return for REINFORCE and 
                        value function for actor critic
       critic_enabled - (optional) whether the baselines are computed by a critic
                      - BxL pytorch tensor Variable (or) scalar (average baseline)
       critic_coeff   - scaling factor for critic loss
       entropies      - (optional) entropies computed at each time step over the action distribution
                      - BxL pytorch Variable
       entropy_coeff  - scaling factor for entropy term in the loss
    """
    L = q_values.size(1)
    batch_size = q_values.size(0)
    # Compute advantage, disconnect baselines from the source of computation
    if critic_enabled:
        adv = Variable(q_values - baselines.data, requires_grad=False)
    else:
        adv = Variable(q_values - baselines, requires_grad=False)

    # Compute policy gradient loss
    loss = - adv * log_probs
    #assert(loss.size() == torch.Size([batch_size, L]))
    loss = loss.sum() / batch_size
    # Compute critic updates if enabled
    if critic_enabled: 
        critic_loss = ((Variable(q_values, requires_grad=False) - baselines)**2).sum() / batch_size
        loss = loss + critic_coeff * critic_loss
    # Compute entropy loss if enabled
    if entropies is not None:
        loss = loss - entropy_coeff * entropies.sum() / batch_size

    return loss
