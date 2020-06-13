[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# DRLND Project 1: Navigation

### Introduction

Presented project is first of three projects required to finish Deep Reinforcement Learning Nanodegree Program [more about program here](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 

All programs that are part of above nanodegree could be found in offical repository: [here](https://github.com/udacity/deep-reinforcement-learning) - inluding starting template for this project [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) that was used to finish presented project (that inluce notebook, code and even part of orginal redme).

### Project Goal and Environment Description

Project goal is to learn agent to navigate in a square world while collecting yellow bananas and avoiding blue bananas. Because task is solved with Reinforcement Learning - agent has to take actions (move) in envirment (square world) in order of maximize cumulative reward (collect as many yellow bananas as possible) and learn by doing.  No pretrained model or traning data with correct behavior is provided for agent - in contrast to Supervised Learning.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, agent must get an average score of +13 over 100 consecutive episodes (requirment for strating that the environment is solved).

The project environment is implemented with The Unity Machine Learning Agents Toolkit (ML-Agents [project repository](https://github.com/Unity-Technologies/ml-agents)) - an open-source project that enables games and simulations to serve as environments for training intelligent agents. Banana environment similar to, but not identical to the Food Collector environment on [the Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

### Getting Started

1. Clone this repository with git command: "git clone https://github.com/TeoDark/Udacity_RL.git" than enter "Project 1" folder.

2. Create or update python environment to match libraries used in project (this step can be easily done with creating anaconda python environment using comand: "conda env create -f environment.yml" with environment.yml provived in project files [more about managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
Enviroment is a bit different than provided by Unity for example it has also Importlib for reloading code files.

2. Download the Unity environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

3. Place the Unity environment file in the DRLND GitHub repository, in the folder within project, and unzip (or decompress) the file. 

4. Run code cell by cell in Navigation.ipynb to recreate experiment and learn DQN to solve Banana environment.

### Raport

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

### Future Ideas

- Better parameter optimalization - using Random Search for finding final hyperparameters could lead to solving the environment in less episodes.
- Implementing different Neural Networks architectures and tweaking with number of layers and neurons in each layer could have impact on learning performance.
- DQN implemented in this project don't implement all six improvements of Rainbow [paper](https://arxiv.org/abs/1710.02298) so implementing all of them should lead to better learning performance.

### References

- [1] https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
- [2] https://github.com/Unity-Technologies/ml-agents
- [3] https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation
- [4] https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn
- [5] Hessel, Matteo, et al. "Rainbow: Combining improvements in deep reinforcement learning." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.