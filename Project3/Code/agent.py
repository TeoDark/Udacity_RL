from Code.neural_network_models import ValueNetwork, SoftQNetwork, PolicyNetwork
from Code.replay_buffer import ReplayBuffer
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

# Refactored code based on: https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb

class Agent():
    """ Agent - class that interact with the environment (act). It stores: 
    a) neural networks allowing the agent to respond by action on state of the environment (act)
    b) variables needed for the learning process of the networks - such as hyperparameters and optimizers (learn). """

    def __init__(self, hyperparameters): 
        """ Constructor - need learning_process_hyperparameters class as input. 
        Initializes neural networks, optimizers, replay buffer and stores hyperparameters. """
        self.hyperparameters = hyperparameters
        self.value_net        = ValueNetwork(num_inputs = hyperparameters.input_size,
                                            hidden_size = hyperparameters.neurons_in_hidden_layer,
                                            init_w = hyperparameters.init_w).to(hyperparameters.device)
        self.target_value_net = ValueNetwork(num_inputs = hyperparameters.input_size,
                                            hidden_size = hyperparameters.neurons_in_hidden_layer,
                                            init_w = hyperparameters.init_w).to(hyperparameters.device)
        self.soft_q_net = SoftQNetwork(num_inputs = hyperparameters.input_size, 
                                    num_actions = hyperparameters.output_size, 
                                    hidden_size = hyperparameters.neurons_in_hidden_layer,
                                    init_w = hyperparameters.init_w).to(hyperparameters.device)
        self.policy_net = PolicyNetwork(num_inputs =hyperparameters.input_size, 
                                    num_actions = hyperparameters.output_size, 
                                    hidden_size = hyperparameters.neurons_in_hidden_layer,
                                    init_w=hyperparameters.init_w, 
                                    log_std_min=hyperparameters.log_std_min, 
                                    log_std_max=hyperparameters.log_std_max).to(hyperparameters.device)

        self.value_criterion  = hyperparameters.value_criterion
        self.soft_q_criterion = hyperparameters.soft_q_criterion
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=hyperparameters.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=hyperparameters.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=hyperparameters.policy_lr)

        self.replay_buffer = ReplayBuffer(capacity = hyperparameters.buffer_capacity , 
                                        sample_batch_size = hyperparameters.sample_batch_size)
    
    def act(self,state):
        """ Method that returns agent action based on current environment state. """
        return self.policy_net.get_action(state, device = self.hyperparameters.device)
    
    def save_network(self, path):
        """ Method stores agent (all networks and hyperparameters) as .pth file in path location. """
        params_attrs = vars(self.hyperparameters)
        all_params = {}
        for item in params_attrs.items():
            all_params[item[0]]=item[1]

        agent_state = {"value_net":self.value_net.state_dict(),
                    "target_value_net":self.target_value_net.state_dict(),
                    "soft_q_net":self.soft_q_net.state_dict(),
                    "policy_net":self.policy_net.state_dict(),
                    "hyperparameters":all_params}
        torch.save(agent_state, str(path))
    
    def learning_step(self):
        """ Method that samples replay buffer and then update agent's neural networks with SAC algorithm.
        A key method to gradually improve agent behaviour by traning process. """

        state, action, reward, next_state, done = self.replay_buffer.sample()
        
        device = self.hyperparameters.device
        gamma = self.hyperparameters.gamma 
        mean_lambda = self.hyperparameters.mean_lambda
        std_lambda = self.hyperparameters.std_lambda
        z_lambda = self.hyperparameters.z_lambda
        soft_tau = self.hyperparameters.soft_tau

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        
        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()


        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda  * log_std.pow(2).mean()
        z_loss    = z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        
        