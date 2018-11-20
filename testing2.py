import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/Deep%20Q%20learning%20with%20Doom.ipynb
env = gym.make('CartPole-v0')
env.reset()
sess = tf.InteractiveSession()

observations = tf.placeholder(tf.float32, shape=[None, 4])  # input params from gym cartpole-v0
actions = tf.placeholder(tf.float32, shape=[None, 2]) #placeholder for action space (left , right)
# y_ = tf.placeholder(tf.float32, shape=[None, 765675])  #
Q_target = tf.placeholder(tf.float32, shape=[None])

# stock 5 h-layer arbitrary neuron count
layer1 = tf.layers.dense(observations, 4, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer2 = tf.layers.dense(layer1, 100, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer3 = tf.layers.dense(layer2, 100, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer4 = tf.layers.dense(layer3, 100, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer5 = tf.layers.dense(layer4, 100, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)

output = tf.layers.dense(layer5, 2, tf.nn.softplus) #value for left and for right

Q = tf.reduce_sum(tf.multiply(output, actions), axis=1)            # Q is our predicted Q value.

loss = tf.reduce_mean(tf.square(Q_target-Q))

optimizer = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())


for _ in range(1000): #epochs
    env.reset()
    rewards = []
    actions = [] #blank lists for values
    observations = []
    obs = env.reset #environment initial state

    for each in range(201): # iterate through frames in environment
        # obs = np.reshape(obs, (-1, 4))
        action = 1 #fill in action loic here
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        observations.append(obs)
        if done:
            break

#compute discounted rewards
#feed actions:rewards to Q network
#update gradients
#add batching?
print(rewards)


# for i in range(10):
#     sess.run(train_step)
#
# def defuzz(fuzz):
#     if fuzz < .5:
#         return 0
#     else:
#         return 0


# def run_episode(env):
#     obs = env.reset()
#     obs = np.reshape(obs,(-1,4))
#     totalreward = 0
#     print(y.eval(feed_dict = {x: obs}))
#     action_fuzz = y.eval(feed_dict = {x: obs}) #run observation through network
#
#     action = defuzz(action_fuzz)
#
#     for _ in range(200):
#         obs, reward, done, info = env.step(action)
#         obs = np.reshape(obs, (-1, 4))
#
#         action_fuzz = y.eval(feed_dict = {x: obs})  # run observation through network _ repeated for each frame
#
#         action = defuzz(action_fuzz)
#
#         totalreward += reward
#         if done:
#             break
# return tf.convert_to_tensor(totalreward)
#
# with tf.name_scope('cross_entropy'):
#     cross_entropy = tf.multiply(-1.0, tf.matmul(Q,Q))  # run_episode(env))
#
#
# sess.run(tf.global_variables_initializer())
#
# for i in range(100):
#
#     obs = env.reset()
#     obs = np.reshape(obs, (-1, 4))
#     totalreward = 0
#
#     action_fuzz = y.eval(feed_dict={x: obs})  # run observation through network
#
#     action = defuzz(action_fuzz)
#
#     for _ in range(201):
#         obs, reward, done, info = env.step(action)
#         obs = np.reshape(obs, (-1, 4))
#
#         action_fuzz = y.eval(feed_dict={x: obs})  # run observation through network _ repeated for each frame
#         print(action_fuzz)
#         action = defuzz(action_fuzz)
#
#         totalreward += 1
#
#         if done:
#             break
#     totalreward = np.reshape(totalreward, (-1, 1))
#     print(totalreward)
#     sess.run(train_step, feed_dict={x: obs, Q: totalreward})

