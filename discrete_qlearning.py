import gym
import numpy as np
import time
from discrete_utility import *



rewards_in_1episode = []
rewards_records = {'episode_index':[],'average_reward':[],'min':[],'max':[]}

for episode in range(EPISODES):
    reward_in_1episode = 0
    if episode % VISUALIZE_IN_EVERY == 0:
        print("episode: ",episode)
        render = True
    else:
        render = False

    state = env.reset()
    done = False
    while not done:
        action = take_epilon_greedy_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        reward_in_1episode += reward

        if render:
            env.render()

        if not done:
            #QLearning
            td_target = reward + DISCOUNT * np.max(q_table[get_discrete_state(next_state)])
            q_table[get_discrete_state(state)][action] += LEARNING_RATE * (td_target - q_table[get_discrete_state(state)][action])
        elif next_state[0] >= 0.5:
            q_table[get_discrete_state(state)][action] = 0
        state = next_state

    #epsilon decaying
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    rewards_in_1episode.append(reward_in_1episode)

    if episode % VISUALIZE_IN_EVERY == 0:
        avg_reward = sum(rewards_in_1episode[-VISUALIZE_IN_EVERY:]) / len(rewards_in_1episode[-VISUALIZE_IN_EVERY:])
        rewards_records['episode_index'].append(episode)
        rewards_records['average_reward'].append(avg_reward)
        rewards_records['min'].append(min(rewards_in_1episode[-VISUALIZE_IN_EVERY:]))
        rewards_records['max'].append(max(rewards_in_1episode[-VISUALIZE_IN_EVERY:]))

ploting(rewards_records,'QLearning in Discrete state','disc_ql')
np.save('disc_ql_qtable.npy',q_table)

done = False
state = env.reset()
while not done:
    action = np.argmax(q_table[get_discrete_state(state)])
    next_state, _, done, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(0.05)

env.close()