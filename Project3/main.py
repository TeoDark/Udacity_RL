from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import Code.agent as ag
import torch
import matplotlib.pyplot as plt

class learning_process_hyperparameters():
    """ Convenient class that stores all parameters and hyperparameters needed for agent training. """
    
    def __init__(self):
        """ Constructor that initializes all parameters and hyperparameters to default values. """
        # general episodes params:
        self.env_path = "tennis_unity_env/Tennis.exe"
        self.max_episodes=2000
        self.score_window_length=100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_score_in_window_to_stop_criterion = 0.6
        self.udacity_mean_score_in_window_required = 0.5

        # replay buffer params:
        self.buffer_capacity = 10000 # 100000
        self.sample_batch_size = 256 # 1024
        
        # optimizer params:
        self.value_criterion = torch.nn.MSELoss()
        self.soft_q_criterion = torch.nn.MSELoss()
        self.value_lr = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4

        # learning step params:
        self.gamma=0.99
        self.mean_lambda=1e-3
        self.std_lambda=1e-3
        self.z_lambda=0.0
        self.soft_tau=1e-2

        # neural network inicialization params:
        self.neurons_in_hidden_layer = 512
        self.init_w = 3e-3
        self.log_std_min=-20
        self.log_std_max=2
        

def main_learning_loop(env, learning_args = None):
    """ Function that trains agent with SAC agorithm on provided environment. 
    Takes unityagents.UnityEnvironment and learning_process_hyperparameters instances as input.
    Returns traned agent and scores from each episode of traning."""

    if(learning_args == None):
        learning_args = learning_process_hyperparameters()

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    agent_number = states.shape[0]
    learning_args.input_size = states.shape[1]
    learning_args.output_size = brain.vector_action_space_size

    print('There are {} agents. Each observes a state with length: {}'.format(agent_number, learning_args.input_size))
    print("Each agent has to make {} actions.".format(action_size))
    print('The state for the first agent looks like:', states[0])


    scores_window = deque(maxlen=learning_args.score_window_length)
    scores_total = []
    agent = ag.Agent(learning_args)

    print("Traning starts!")
    for episode in range(1, learning_args.max_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations 
        score = np.zeros(num_agents)

        while True:
            actions = agent.act(state)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards 
            dones = env_info.local_done

            # separate update of replay_buffer for each agent:
            for i in range(agent_number):
                agent.replay_buffer.push(state[i],actions[i],rewards[i],next_states[i],dones[i])

            # agent is traned only if replay buffer has more samples than batch size:
            if(len(agent.replay_buffer)>=agent.replay_buffer.sample_batch_size):
                agent.learning_step()

            score += rewards
            state = next_states
            
            if np.any(dones):
                break

        score_in_episode = np.max(score)
        scores_total.append(score_in_episode)
        scores_window.append(score_in_episode)
        mean_score = np.mean(scores_window)

        print('\rEpisode {}\tEpisode Score:{:.10f}\tMean in window:{:.10f}'.format(episode,score_in_episode,mean_score), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tEpisode Score:{:.10f}\tMean in window:{:.10f}'.format(episode,score_in_episode,mean_score))
            agent.save_network("Checkpoints/agent_tmp_{}.pth".format(episode))
        if mean_score >= learning_args.udacity_mean_score_in_window_required:
            print('\nEnvironment solved in {} episodes!\tMean in window: {:.10f} But we keep going for better score!'.format(episode,mean_score))
        if mean_score >=learning_args.mean_score_in_window_to_stop_criterion: # for sure i use a bit larger score than required
            print('\nEnvironment reach stop criterion in {} episodes!\tMean in window: {:.10f}'.format(episode,mean_score))    
            agent.save_network("Checkpoints/final_agent_done_in_{}.pth".format(episode))
            break
    
    print("Traning ends!")
    return agent, scores_total

def create_plot_and_save_to_file(scores_total, score_window_length, plot_filename="Images/learning_process.png"):
    """ Function that plot figure from provided scores and saves such figure as png file. 
    Takes scores array, window_length for calculating mean in window and filename to save figure to designated location. """

    plot_score_mean = []
    plot_score_in_window = []
    plot_score_in_window_que = deque(maxlen=100)

    for s in scores_total:
        plot_score_mean.append(np.mean(s))
        plot_score_in_window_que.append(np.mean(s))
        plot_score_in_window.append(np.mean(plot_score_in_window_que))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(plot_score_mean)), plot_score_mean, color='b', label='episode score')
    ax.plot(np.arange(len(plot_score_in_window)), plot_score_in_window, color='r', label='ep mean score in window')
    ax.legend()
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(plot_filename)
    plt.show()


def observe_learned_agent(env, agent, replays = 10):
    """ Function that run trained agent in environment - let user see agent behaviour as simmulation is in "normal" time speed.
    Takes unityagents.UnityEnvironment, agent and number of replays as input. """

    print("Observing trained agent:")
    brain_name = env.brain_names[0]
    for i in range(1, replays+1):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations 
        num_agents = len(env_info.agents)
        scores = np.zeros(num_agents)
                
        while True:
            actions = agent.act(state)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards 
            dones = env_info.local_done
            scores += rewards
            state = next_states

            if np.any(dones):
                break
        print('Score from episode {}: {:.10f}\t scores per agent: {}'.format(i, np.max(scores), scores))

if __name__ == "__main__":
    """ Function code that will be ran only when file is running as the primary module.
    It does everything in sense of traning agent with SAC algorithm on Tennis environment.
    It starts with creating default hyperparameters then creating environment then traning agent then
    creating traning process plot then observing trained agent and finally closing environment. """

    hyperparameters = learning_process_hyperparameters()
    # hyperparameters.max_episodes = 200 # for fast loop testing

    env = UnityEnvironment(file_name= hyperparameters.env_path)

    agent, scores_total = main_learning_loop(env = env, learning_args=hyperparameters)

    # for saving plot of learning process:
    create_plot_and_save_to_file(scores_total, hyperparameters.score_window_length)

    # for observing agent after learning:
    observe_learned_agent(env = env, agent = agent)

    env.close()

