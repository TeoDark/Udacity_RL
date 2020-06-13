
# DRLND Project 3: Collaboration and Competition

## Introduction

Presented project is third of three projects required to finish Deep Reinforcement Learning Nanodegree Program - [more about program here](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 

All programs that are part of above nanodegree could be found in offical repository: [here](https://github.com/udacity/deep-reinforcement-learning) - inluding starting template for this project [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) that was used to finish presented project (that inluce notebook, code and even part of orginal redme).

Agent solving environment is trained with Soft Actor-Critic Algorithm (SAC - [more about algorithm](https://arxiv.org/abs/1801.01290)). Implementation used in project is almost completely based on code from [this repository](https://github.com/higgsfield/RL-Adventure-2) with only few adjustments.

## Project Goal and Environment Description

The project environment is implemented with The Unity Machine Learning Agents Toolkit (ML-Agents [project repository](https://github.com/Unity-Technologies/ml-agents)) - an open-source project that enables games and simulations to serve as environments for training intelligent agents. Udacity's Tennis environment is similar to, but not identical to the Unity's Tennis environment that can be find on [the Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores - this yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

![Alt Text](./Images/after.gif)

## Getting Started

Inspect 'Report.md' for detail about learning process and experiment.

Run 'tennis.ipynb' cell by cell to re-create experiment.

## Installation guide

1. Clone this repository with git command: "git clone https://github.com/TeoDark/Udacity_RL.git" than enter "Project3" folder.

2. Create or update python environment to match libraries used in project (this step can be easily done with creating anaconda python environment using comand: "conda env create -f environment.yml" with environment.yml provived in project files [more about managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
Enviroment is a bit different than provided by Unity for example it has also Importlib for reloading code files.

2. Download the Unity environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    (_For Windows users_) 
    
    Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

3. Place the unziped Unity environment file in the folder: "tennis_unity_env"  within project. 

4. Run code cell by cell in tennis.ipynb to recreate experiment and learn SAC to solve Tennis environment.

## References

- [1] https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
- [2] https://github.com/Unity-Technologies/ml-agents
- [3] https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet
- [4] https://arxiv.org/abs/1801.01290
- [5] https://github.com/higgsfield/RL-Adventure-2
