import matplotlib.pyplot as plt
import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODES = 10000
VISUALIZE_IN_EVERY = 50
Q_TABLE_LENGTH = 40
#decaying epsilon from episode 1 to episodes/2
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2

discrete_size = [Q_TABLE_LENGTH] * len(env.observation_space.high)
discrete_window_size = (env.observation_space.high - env.observation_space.low) / discrete_size
q_table = np.zeros(discrete_size + [env.action_space.n])
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

def get_discrete_state (state):
    discrete_state = (state - env.observation_space.low) // discrete_window_size
    return tuple(discrete_state.astype(int))

def take_epilon_greedy_action(state, epsilon):
    discrete_state = get_discrete_state(state)
    if np.random.random() < epsilon:
        action = np.random.randint(0,env.action_space.n)
    else:
        action = np.argmax(q_table[discrete_state])
    return action

def get_policy_expectation(next_state,epsilon):
    discrete_state = get_discrete_state(next_state)
    exp = 0
    exp += (1-epsilon)*np.max(q_table[discrete_state])
    for i in range(0,env.action_space.n):
        exp += (epsilon/env.action_space.n)*q_table[discrete_state][i]
    return exp

def ploting(rewards_records,title,filename):
    plt.plot(rewards_records['episode_index'], rewards_records['average_reward'], label = 'average reward')
    plt.plot(rewards_records['episode_index'], rewards_records['min'], label = 'min reward')
    plt.plot(rewards_records['episode_index'], rewards_records['max'], label = 'max reward')
    plt.legend(loc='upper left')
    plt.xlabel('Episodes Index')
    plt.ylabel('Reward')
    plt.title(title)
    plt.savefig(filename+'.jpg') 