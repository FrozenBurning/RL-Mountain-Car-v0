from continuous_utility import *
import time
q_table = np.load('cont_ql_qtable.npy')
# env = gym.wrappers.Monitor(env, "recording")

done = False
state = env.reset()
while not done:
    action_idx = np.argmax(q_table[get_discrete_state(state)])
    action = action_space[action_idx]
    next_state, _, done, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(0.05)

env.close()