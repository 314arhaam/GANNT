import gym
import time
import math
import numpy as np

n_population = 40
n_generation = 6
n_input = 4
n_hidden_layer = 3
n_output = 1
env = gym.make('CartPole-v0')

def sigmoid(x, b):
    return 1/(1 + math.exp(b-x))

def heaviside(x, b):
    return x >= b and 1 or 0

def fun(input, weight, bias):
    y = 0
    for j in range(n_hidden_layer):
        y += sigmoid(sum(weight[j * n_input + i] * input[i] for i in range(n_input)), bias[j]) * weight[n_hidden_layer * n_input + j]
    y = heaviside(y, bias[n_hidden_layer])
    return y


def ramp(t, n = 20):
	if not t % n:
		env.step(1)

def main():
	maxTime = 1500
	env._max_episode_steps = maxTime
	weight = [0.609703628,	0.166980294,	1.035782004,	0.715189222,	0.302665038,	0.161967064,	0.946068623,	0.041790416,	0.284546934,	0.540528025,	0.550177066,	0.559302551,	0.228917777,	0.168908509,	0.951744706]
	bias = [0.978349763,	0.018172706,	0.557961361,	0.480916889]
	observation = env.reset()
	for t in range(1500):
	    env.render()
	    action = fun(observation, weight, bias)
	    observation, reward, done, info = env.step(action)
	    #ramp(t, n=10) #done must be included, things must be returned from ramp function
	    if done:
	        print("Episode finished after {} timesteps".format(t+1))
	        break
	env.close()

if __name__ == '__main__':
	main()
