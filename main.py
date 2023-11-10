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
#

import joblib
import keras
import tensorflow as tf
from matplotlib import pyplot as plt, animation
import gym
import pyvirtualdisplay
import numpy as np


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

class TrainingExercises:
    def __init__(self):
        print("ok1")

    def ex1(self):
        env = gym.make("CartPole-v1", render_mode='rgb_array')
        obs = env.reset()
        # print(obs)

        display = pyvirtualdisplay.Display(visible=False, size=(1400, 900)).start()

        plot_environment(env)
        # plt.show()
        # print(env.action_space)
        action = 1  # accelerate right

        obs, reward, done, info, dummy = env.step(action)
        print(obs, reward, done, info, dummy)


def program1():
    TrainingExercises()


if __name__ == '__main__':
    program1()
