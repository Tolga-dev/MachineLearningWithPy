# reinforcement learning
# its the most exciting fields of ml today, in games particularly.
# TD-Gammon, back-gammong playing program.

# TD gammon
# temporal difference learning is an approach to learning how to predict a quantity
# that depends on future values of a given signal.
# its combination of monte carlo and dynamic programming

# techniques,
# q networks, markov decision processes

# learning to optimize Rewards
# Policy Search
# algos a software agent uses to determine its actions is called its policy
# random deterministic env is stochastic policy

# open ai gym
# we need to have a working env.

# neural network policies

# evaluating actions, the credit assignment problem

# policy gradients
# pg algorithms optimize the parameters of a policy by following the gradient toward higher rewards
# popular one is 'reinforce' algorithms
# DISCRETE DOMAIN + CONTINUOUS DOMAIN
# policy-based
# on-policy + off-policy
# model free

# markov decision processes
# they are used heavily used in thermodynamics, chemistry, statics, and mush more

# Q learning
# it is an adaptation of the q-value iteration algorithm to the situation where
# transition probabilities and the rewards are initially unknown.
# It is called an off-policy algorithm, since policy being trained is not necessarily the one being executed

# exploration Policies
# visiting everything may take times a lot, so we can use e-greedy policy

# deep q-learning variants
# some algorithms that can stabilize and speed up training

# fixed q-value targets
# in the basic deep q-learning algorithm, the model is used to make prediction and set its own targets

# double dqn

# dueling dqn

# TF agent lib
# reinforcement learning lib, based on tensorflow

# Overview of some popular RL algorithms
# actor critic
#


import joblib
import keras
import tensorflow as tf
from matplotlib import pyplot as plt, animation
import pyvirtualdisplay
import numpy as np
import gymnasium as gym
import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
from functools import partial
from gym.wrappers import TimeLimit
from tf_agents.environments import suite_atari, TFPyEnvironment
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

from tf_agents.environments import suite_gym, wrappers, ActionRepeat

train_model_cache_download = joblib.Memory('./tmp/ReinforcementLearning/train_model_cache_download')


@train_model_cache_download.cache
def getDataImdbFrom():
    pass


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()


def plot_environment(env, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render()
    plt.imshow(img)
    plt.axis("off")
    return img


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim




max_episode_steps = 27000  # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "BreakoutNoFrameskip-v4"


class AtariPreprocessingWithAutoFire(AtariPreprocessing):
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        super().step(1)  # FIRE to start
        return obs

    def step(self, action):
        lives_before_action = self.ale.lives()
        obs, rewards, done, info = super().step(action)
        if self.ale.lives() < lives_before_action and not done:
            super().step(1)  # FIRE to start after life lost
        return obs, rewards, done, info


def plot_observation(obs):
    # Since there are only 3 color channels, you cannot display 4 frames
    # with one primary color per frame. So this code computes the delta between
    # the current frame and the mean of the other frames, and it adds this delta
    # to the red and blue channels to get a pink color for the current frame.
    obs = obs.astype(np.float32)
    img = obs[..., :3]
    current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img / 150, 0, 1)
    plt.imshow(img)
    plt.axis("off")


class TrainingExercises:
    def __init__(self):
        print("ok1")

        env = suite_gym.load("Breakout-v4")
        # env.seed(42)
        # env.reset()
        # env.step(1)  # Fire
        # img = env.render(mode="rgb_array")

        # plt.figure(figsize=(6, 8))
        # plt.imshow(img)
        # plt.axis("off")
        # plt.show()

        # Environment Specifications
        # print(env.current_time_step())
        # print(env.observation_spec())
        # print(env.action_spec())
        # print(env.time_step_spec())
        # Environment Wrappers
        # print(env.observation_spec())
        # print(env.action_spec())
        # print(env.time_step_spec())
        # print(env.gym.get_action_meanings())
        # to create a wrapped environment, we must create a wrapper.
        # repeating_env = ActionRepeat(env, times=4)
        # limited_repeating_env = suite_gym.load(
        #     "Breakout-v4",
        #     gym_env_wrappers=[partial(TimeLimit, max_episode_steps=10000)],
        #     env_wrappers=[partial(ActionRepeat, times=4)],
        # )

        # env = suite_atari.load(
        #     environment_name,
        #     max_episode_steps=max_episode_steps,
        #     gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4])
        # env.seed(42)
        # env.reset()
        # time_step = None
        #
        # for _ in range(4):
        #     time_step = env.step(3)  # LEFT
        # plt.figure(figsize=(6, 6))
        # plot_observation(time_step.observation)
        # plt.show()

        # make env usable from withing a tensorflow grph
        # tf_env = TFPyEnvironment(env)

class GeneralReinforcement:
    def __init__(self):
        env = gym.make("CartPole-v1")
        env.action_space.seed(42)
        env.reset(seed=42)
        np.random.seed(42)

        # markov probabilities
        transition_probabilities = [  # shape=[s, a, s']
            [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
            [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
            [None, [0.8, 0.1, 0.1], None]]
        rewards = [  # shape=[s, a, s']
            [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
            [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]
        possible_actions = [[0, 1, 2], [0, 2], [1]]

        def ex0():
            obs = env.reset()
            print(obs)
            img = env.render()
            print(img.shape)
            print(env.action_space)

            action = 1
            obs, reward, done, info, dummy = env.step(action)
            print(obs, reward, done, info, dummy)

            totals = []
            for episode in range(500):
                episode_rewards = 0
                obs, info = env.reset()
                for step in range(200):
                    action = basic_policy(obs)
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_rewards += reward
                    if terminated:
                        break
                totals.append(episode_rewards)

            print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

        def ex1():
            frames = []
            obs, info = env.reset()
            for step in range(20000):
                img = env.render()
                frames.append(img)
                action = basic_policy(obs)
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    break
            print(len(frames))
            plot_animation(frames).save('animation.mp4', writer='ffmpeg', fps=30)
            env.close()

        def ex2():
            keras.backend.clear_session()
            tf.random.set_seed(42)
            np.random.seed(42)

            n_inputs = 4  # == env.observation_space.shape[0]

            model = keras.models.Sequential([
                keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
                keras.layers.Dense(1, activation="sigmoid"),
            ])
            frames = []
            obs, info = env.reset()
            for step in range(200):
                frames.append(env.render())
                left_proba = model.predict(obs.reshape(1, -1))[0][0]
                print(left_proba)
                action = int(np.random.rand() > left_proba)
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    break
            print(len(frames))
            plot_animation(frames).save('animation.mp4', writer='ffmpeg', fps=30)
            env.close()

        def ex3():
            def step(state, action):
                probas = transition_probabilities[state][action]
                next_state = np.random.choice([0, 1, 2], p=probas)
                reward = rewards[state][action][next_state]
                return next_state, reward

            def exploration_policy(state):
                return np.random.choice(possible_actions[state])

            np.random.seed(42)

            Q_values = np.full((3, 3), -np.inf)
            for state, actions in enumerate(possible_actions):
                Q_values[state][actions] = 0

            alpha0 = 0.05  # initial learning rate
            decay = 0.005  # learning rate decay
            gamma = 0.90  # discount factor
            state = 0  # initial state
            history2 = []  # Not shown in the book

            for iteration in range(10000):
                history2.append(Q_values.copy())  # Not shown
                action = exploration_policy(state)
                next_state, reward = step(state, action)
                next_value = np.max(Q_values[next_state])  # greedy policy at the next step
                alpha = alpha0 / (1 + iteration * decay)
                Q_values[state, action] *= 1 - alpha
                Q_values[state, action] += alpha * (reward + gamma * next_value)
                state = next_state
            print(Q_values)
            print(np.argmax(Q_values, axis=1))


def program1():
    TrainingExercises()


if __name__ == '__main__':
    program1()
