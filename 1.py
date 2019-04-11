import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
global GAMMA
GAMMA = 0.9  # discount factor
INITIAL_EPSILON = 0.6  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
EPSILON_DECAY_STEPS = 100  # decay period
HIDDEN_NODES = 20

REPLAY_SIZE = 10000  # experience replay buffer size

BATCH_SIZE = 200  # size of minibatch

global replay_buffer  # 这个buffer是global的

replay_buffer = []
# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
w1 = tf.Variable(tf.random_normal(shape=[STATE_DIM, HIDDEN_NODES], seed=1), collections=c_name)

b1 = tf.Variable(tf.random_normal(shape=[1, HIDDEN_NODES], seed=1), collections=c_name)

output_layer1 = tf.nn.relu(tf.matmul(state_in, w1) + b1)

# value

w2 = tf.Variable(tf.random_normal(shape=[HIDDEN_NODES, 1], seed=1), collections=c_name)

b2 = tf.Variable(tf.random_normal(shape=[1, 1], seed=1), collections=c_name)

value = tf.matmul(output_layer1, w2) + b2

# advantage

w3 = tf.Variable(tf.random_normal(shape=[HIDDEN_NODES, ACTION_DIM], seed=1), collections=c_name)

b3 = tf.Variable(tf.random_normal(shape=[1, ACTION_DIM], seed=1), collections=c_name)

advantage = tf.matmul(output_layer1, w3) + b3

out = tf.nn.relu(value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)))

# TODO: Network outputs
q_values = out
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_sum(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
train_loss_summary_op = tf.summary.scalar("TrainingLoss", loss)
# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):  # get action
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


count = 0
# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        one_hot_action = np.zeros(ACTION_DIM)

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > REPLAY_SIZE:
            replay_buffer.pop(0)

        # Do one training step
        if (len(replay_buffer) > BATCH_SIZE):
            minibatch = random.sample(replay_buffer, BATCH_SIZE)  # 随机的在buffer中获取 bitchsize长度的数据

            state_batch = [data[0] for data in minibatch]

            action_batch = [data[1] for data in minibatch]

            reward_batch = [data[2] for data in minibatch]

            next_state_batch = [data[3] for data in minibatch]

            target_batch = []

            Q_value_batch = q_values.eval(feed_dict={

                state_in: next_state_batch

            })

            for i in range(0, BATCH_SIZE):

                sample_is_done = minibatch[i][4]

                if sample_is_done:

                    target_batch.append(reward_batch[i])

                else:

                    # TO IMPLEMENT: set the target_val to the correct Q value update

                    target_val = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])

                    target_batch.append(target_val)
            summary, _ = session.run([train_loss_summary_op, optimizer], feed_dict={
                target_in: target_batch,
                state_in: state_batch,
                action_in: action_batch
            })
            count += 1

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                # env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
