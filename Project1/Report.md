### Raport

(Same as in Raport section in README.md)

Agent model is using Deep Q-Network(DQN) that is implemented in Python with PyTorch framework (Code with some modifications is close to identical to code provided by Udacity for lesson: Deep Q-Networks for solving OpenAI Gym's LunarLander-v2 environment [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn)).

The DQN has 3 fully connected layers with ReLU activation function. Network input is set for number of enviroment states space - 37, than both first and second layer have 64 neurons in each layer. Last third layer has 4 neurons - one for each action that agent could perform. (DQN use default parameters values from it's constructor in model.py file)

Implemented DQN has two improvements over classic DQN: A) Experience Replay - past model experiences are saved in buffer as tuples of (state, action, reward, next_state) and random batches of experiences are used in learning process that allows to learn more robust policy, not affected by the inherent correlation present in the sequence of observed experience tuples.  B) Fixed Target - Two sets of DQN are used in learning process (same architecture) one for estimating true value function (target) and second for currently predicted Q-Value (local) we keep target network weight fixed and update it after updating local network weights - dequpling the target from the parameters leads to more stable and less likely to diverge or fall into oscillations learning process.

After few attepmts to modyfy network in DQN (for example by adding Dropout) and randomly switching epsilon related hyperparameters (epsilon decay and minimal epsilon value), which failed. Short grid search with these easily accessible parameters was conducted - same values we were tested 10 times to make process a bit more robust. To limit comutation in grid search only 100 were computed per parameters tuple. After such grid search promissing parameters were choosen to final DQN learning process that resulted in solving the environment in 332 episodes.

Hyperparameters choosen for final DQN learning process:  

A) Hardcoded inside dqn_agent.py file:
- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 64         # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR = 5e-4               # learning rate 
- UPDATE_EVERY = 4        # how often to update the network 

B) Set in Navigation.ipynb:
- MAX_STEPS_IN_EPOCH = 300 # maximum number of iterations in single epoch
- eps_start=1.0 # starting value of epsilon in first epoch 
- eps_end=0.05  # minimum value of epsilon that cannot be dacayed
- eps_decay=0.95 # value of epsilon decay each epoch