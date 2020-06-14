from collections import deque
import numpy as np
import random

# Refactored code based on: https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb

class ReplayBuffer:
    """ Class that stores previous agent experiences to retain memory that can be random sampled for training agent. 
    Such samples are not as correlated as "sequential" memory  - it's beneficial for traning process. """
    
    def __init__(self, capacity = 100000, sample_batch_size = 1024):
        """ Constructor - crate replay buffer of specific size, also sets batch size for future sampling. """
        self.capacity = capacity
        self.sample_batch_size = sample_batch_size
        self.replay_buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """ Method that add memory of single experience (state, action, reward, next_state, done)
         to total stored memory called replay buffer. """
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        """ Method that returns random sample (size of batch size) from total stored memory. """ 
        replay = random.sample(self.replay_buffer, k=self.sample_batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*replay))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """ Method that returns current replay buffer length. """ 
        return len(self.replay_buffer)