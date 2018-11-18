import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make('CartPole-v0')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 4])  # input params from gym cartpole-v0
y_ = tf.placeholder(tf.float32, shape=[None, 1])  # output 0-1 will round to 0 or 1 to determine action
Q = tf.placeholder(tf.float32, shape=[None, 1])

# stock 5 h-layer arbitrary neuron count
layer1 = tf.layers.dense(x, 4, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer2 = tf.layers.dense(layer1, 100, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer3 = tf.layers.dense(layer2, 100, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer4 = tf.layers.dense(layer3, 100, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer5 = tf.layers.dense(layer4, 100, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)

y = tf.layers.dense(layer5, 1, tf.nn.softplus)


# for i in range(10):
#     sess.run(train_step)

def defuzz(fuzz):
    if fuzz < .5:
        return 0
    else:
        return 0


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

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.multiply(-1.0, tf.matmul(Q,Q))  # run_episode(env))

ew
sess.run(tf.global_variables_initializer())

for i in range(100):

    obs = env.reset()
    obs = np.reshape(obs, (-1, 4))
    totalreward = 0

    action_fuzz = y.eval(feed_dict={x: obs})  # run observation through network

    action = defuzz(action_fuzz)

    for _ in range(201):
        obs, reward, done, info = env.step(action)
        obs = np.reshape(obs, (-1, 4))

        action_fuzz = y.eval(feed_dict={x: obs})  # run observation through network _ repeated for each frame
        print(action_fuzz)
        action = defuzz(action_fuzz)

        totalreward += 1

        if done:
            break
    totalreward = np.reshape(totalreward, (-1, 1))
    print(totalreward)
    # sess.run(train_step, feed_dict={x: obs, Q: totalreward})

