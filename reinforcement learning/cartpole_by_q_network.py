import numpy as np
import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


env = gym.make("CartPole-v0")

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# set network
X = tf.placeholder(tf.float32, [None, input_size], name="input_x")
W1 = tf.get_variable("W1", shape=[input_size, output_size], initializer=tf.keras.initializers.he_uniform())

q_predict = tf.matmul(X, W1)

Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - q_predict))

learning_rate = 1e-1
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

num_episodes = 2000
discount_rate = 0.9
result_list = []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        e = 1. / ((i / 10) + 1)
        step_count = 0
        s = env.reset()
        done = False

        while not done:
            step_count += 1

            x = np.reshape(s, [1, input_size])

            Qs = sess.run(q_predict, feed_dict={X: x})

            # get a action
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            # go next step
            s1, reward, done, _ = env.step(a)
            
            if done:
                Qs[0, a] = -100
            else:
                x1 = np.reshape(s1, [1, input_size])
                Qs1 = sess.run(q_predict, feed_dict={X: x1})

                Qs[0, a] = reward + discount_rate * np.max(Qs1)

            # train
            sess.run(train, feed_dict={X: x, Y: Qs})

            s = s1

        result_list.append(step_count)
        print("Episode : {}, steps : {}".format(i, step_count))

        # exit condition
        if len(result_list) > 10 and np.mean(result_list[-10:]) > 500:
            break

    # applying with trained weight
    observation = env.reset()
    reward_sum = 0

    while True:
        env.render()

        x = np.reshape(observation, [1, input_size])
        Qs = sess.run(q_predict, feed_dict={X: x})
        a = np.argmax(Qs)

        observation, reward, done, _ = env.step(a)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break
