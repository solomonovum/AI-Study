import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def one_hot_encoder(state):
    return np.identity(16)[state:state + 1]


env = gym.make("FrozenLake-v0")

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

q_predict = tf.matmul(X, W)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Y - q_predict))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

num_episodes = 2000
r = .99
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_episodes):
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False

        while not done:
            Qs = sess.run(q_predict, feed_dict={X: one_hot_encoder(s)})

            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            s1, reward, done, _ = env.step(a)

            if done:
                Qs[0, a] = reward
            else:
                Qs1 = sess.run(q_predict, feed_dict={X: one_hot_encoder(s1)})
                Qs[0:a] = reward + r * np.max(Qs1)

            sess.run(train, feed_dict={X: one_hot_encoder(s), Y: Qs})
            rAll += reward
            s = s1
        rList.append(rAll)

print("Success Rate : ", str(sum(rList) / num_episodes))
plt.plot(range(len(rList)), rList)
plt.show()
