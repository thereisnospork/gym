import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')


def run_episode(env, parameters):
    obs = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,obs) < 0 else 1
        obs, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


hist = []
for _ in range(10000):
    bestparams = None
    bestreward = 0
    for i in range(10000):
        # print(i)
        params = np.random.rand(4)
        reward = run_episode(env,params)
        if reward > bestreward:
            bestreward = reward
            bestparams = params
            if reward == 200:
                # print(params)
                # print(i)
                hist.append(i)
                break

print(hist)
plt.hist(hist, bins = 30, density = True, rwidth=.95, color = 'blue')
plt.show()

#
# for _ in range(20):
#     observation = env.reset()
#     param = np.random.rand(4)
#     action =
#
#
#
#     for t in range(1000):
#         env.render()
#         # print(observation)
#         param = np.random.rand(4)
#         # print(np.sum(observation))
#         action = env.action_space.sample()
#         # action = 1
#         observation, reward, done, info = env.step(action)
#         print(reward)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break