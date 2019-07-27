# python3 GANNT.py

import gym
import time
import math
import numpy as np
from pandas import DataFrame

class GANNT:
    def __init__(self, numberOfInputs=4, numberOfHiddenLayers=3, numberOfOutputs=1, numberOfGenerations=3, population=20, maxEpisodeSteps=1500):
        self.population = population
        self.n_generation = numberOfGenerations
        self.maxTime = maxEpisodeSteps
        self.n_input = numberOfInputs
        self.n_hidden_layer = numberOfHiddenLayers
        self.n_output = numberOfOutputs
        self.W = 0
        self.B = 0
        self.best = 0

    def sigmoid(self, x, b):
        return 1/(1 + math.exp(b - x))

    def heaviside(self, x, b):
        return x >= b and 1 or 0
    
    def fun(self, inputs, weight, bias):
        y = 0
        for j in range(self.n_hidden_layer):
            y += self.sigmoid(sum(weight[j * self.n_input + i] * inputs[i] for i in range(self.n_input)), bias[j]) * weight[self.n_hidden_layer * self.n_input + j]
        y = self.heaviside(y, bias[self.n_hidden_layer])
        return y

    def sex(self, A, B, n):
        l = len(A)
        gene = np.zeros([n, l])
        noise = np.random.rand(n, l) - 0.5
        gene = 0.5 * (A + B) + 2 * (A - B) * noise
        return gene

    def run(self):
        env = gym.make('CartPole-v0')
        env._max_episode_steps = self.maxTime
        weight = np.random.rand(self.population, self.n_input * self.n_hidden_layer + self.n_hidden_layer * self.n_output)
        bias = np.random.rand(self.population, self.n_hidden_layer + self.n_output)
        action = 0
        means = []
        for generation in range(self.n_generation):
            print("Generation begins")
            time.sleep(1)
            times = np.zeros(self.population)
            for seed in range(self.population):
                name = ''.join([chr(int(i)+65) for i in np.round(np.random.rand(5)*25+1)])
                observation = env.reset()
                for t in range(self.maxTime):
                    env.render()
                    action = self.fun(observation, weight[seed], bias[seed])
                    observation, reward, done, info = env.step(action)
                    if done:
                        print(str(generation) + str(seed).zfill(2) + ' - ' + name + " finished after %3d timesteps"%(t+1))
                        times[seed] = t
                        break
            A, B, *_ = np.argsort(times)[::-1]
            wA = weight[A, :]
            wB = weight[B, :]
            bA = bias[A, :]
            bB = bias[B, :]
            weight = self.sex(wA, wB, self.population)
            bias = self.sex(bA, bB, self.population)
            means += [np.mean(times) + 1]
            print("\033[101m"+"Generation %d time %.3f"%(generation, means[-1])+"\033[0m"+"\n")    
        self.dataFrameMaker(weight, bias, times)
        env.close()
        self.W = weight
        self.B = bias
        self.best = A

    def dataFrameMaker(self, weight, bias, times):
        dictionary = dict()
        for i in range(self.n_input * self.n_hidden_layer + self.n_hidden_layer * self.n_output):
            dictionary['Weight ' + str(i)] = weight[:, i]
        for i in range(self.n_hidden_layer + self.n_output):
            dictionary['Bias ' + str(i)] = bias[:, i]
        dictionary['Time-Steps'] = times
        dictionary = DataFrame(dictionary)
        dictionary.to_excel("GAN_Results.xlsx", sheet_name="Final Generation")

    def __call__(self, num=-1):
        if num != -1:
            num = self.best
        env = gym.make('CartPole-v0')
        env._max_episode_steps = self.maxTime
        observation = env.reset()
        print("Playing...")
        time.sleep(1)
        for t in range(self.maxTime):
            env.render()
            action = self.fun(observation, self.W[num], self.B[num])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        env.close()

    def environmentView(self):
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 200
        observation = env.reset()
        print("Initial State")
        time.sleep(1)
        for t in range(200):
            env.render()
            env.step(env.action_space.sample())
        env.close()


if __name__ == '__main__':
    gen = GANNT()
    gen.environmentView()
    gen.run()
    gen()
