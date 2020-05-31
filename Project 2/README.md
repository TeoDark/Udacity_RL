
# DRLND Project 2: Continuous Control

## Introduction

Presented project is second of three projects required to finish Deep Reinforcement Learning Nanodegree Program - [more about program here](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 

All programs that are part of above nanodegree could be found in offical repository: [here](https://github.com/udacity/deep-reinforcement-learning) - inluding starting template for this project [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control) that was used to finish presented project (that inluce notebook, code and even part of orginal redme).

Agent solving environment is trained with Proximal Policy Optimization Algorithm (PPO - [more about algorithm](https://arxiv.org/pdf/1707.06347.pdf)). Implementation used in project is almost complitly based on code from [this repository](https://github.com/reinforcement-learning-kr/pg_travel) with only few adjustments.

## Project Goal and Environment Description

The project environment is implemented with The Unity Machine Learning Agents Toolkit (ML-Agents [project repository](https://github.com/Unity-Technologies/ml-agents)) - an open-source project that enables games and simulations to serve as environments for training intelligent agents. Udacity's Reacher environment is similar to, but not identical to the Unity's Reacher environment that can be find on [the Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Udacity provide two versions of environment: a) The first version contains a single agent. b) The second version contains 20 identical agents, each with its own copy of the environment. In presented project second one environment was used. The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).

![Alt Text](./Gifs/after.gif)

## Getting Started

Inspect 'Report.md' for detail about learning process and experiment.

Run 'Continuous_Control.ipynb' cell by cell to recreate experiment.

## Installation guide

1. Clone this repository with git command: "git clone https://github.com/TeoDark/Udacity_RL.git" than enter "Project 2" folder.

2. Create or update python environment to match libraries used in project (this step can be easily done with creating anaconda python environment using comand: "conda env create -f environment.yml" with environment.yml provived in project files [more about managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
Enviroment is a bit different than provided by Unity for example it has also Importlib for reloading code files.

2. Download the Unity environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

3. Place the unziped Unity environment file in the folder: "Reacher_Many"  within project. 

4. Run code cell by cell in Continuous_Control.ipynb to recreate experiment and learn PPO to solve Reacher environment.

## References

- [1] https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
- [2] https://github.com/Unity-Technologies/ml-agents
- [3] https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control
- [4] https://arxiv.org/pdf/1707.06347.pdf
- [5] https://github.com/reinforcement-learning-kr/pg_travel