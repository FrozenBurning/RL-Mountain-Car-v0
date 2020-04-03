<!--
 * @Description: 
 * @Author: Zhaoxi Chen
 * @Github: https://github.com/FrozenBurning
 * @Date: 2020-04-03 11:35:25
 * @LastEditors: Zhaoxi Chen
 * @LastEditTime: 2020-04-03 11:44:57
 -->
# Mountain Car v0 & v0-Continuous

**Author: Zhaoxi Chen**

**Feature:**
- Solved Mountain Car-v0 and Mountain Car-v0 Continuous problems
- Using Q-Learning, Sarsa, Exp-Sarsa
- Implement DQN as solution to continuous state space


## Dependecies

Python 3.6+
[gym](https://github.com/openai/gym)

## Usage

python scripts with "continuous_" prefix are solutions to Mountain Car-v0 continuous while "discrete_" prefix imply solutions to Mountain Car-v0

**E.g.** If you want to take Q-learning as solution to Mountain Car-v0, run as the following:
```bash
python3 discrete_qlearning.py
```

All of the scipts will save qtable as .npy and also plot reward curves.