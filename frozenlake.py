import numpy as np
import pandas as pd
import scipy
import time
import sys
import matplotlib.pyplot as plt
import gym
from gym import wrappers
from gym.envs.toy_text import frozen_lake
from myUtil import *

prob_name = "FrozenLake"
np.random.seed(2)
#test different size of map, same as num of states
time_array=[]
iter_array=[]
score_array=[]
mapsize = [2, 5, 8, 16, 32]

for i in mapsize:
    map_size = 8
    my_map = frozen_lake.generate_random_map(size=i, p=0.9)
    env = gym.make("FrozenLake-v1", desc=my_map)
    env = env.unwrapped
    start=time.time()
    best_policy,n_iter = policy_iteration(env, 0.95)
    scores = evaluate_policy(env, best_policy, 0.95)
    end=time.time()
    score_array.append(np.mean(scores))
    iter_array.append(n_iter)
    time_array.append(end-start)

print(mapsize)
print(time_array)
print(score_array)
print(iter_array)
plt.figure()
plt.plot(mapsize,time_array)
plt.xlabel('Map Size')
plt.ylabel('Runtime')
plt.title('Policy Iteration, Runtime')
plt.grid()
plt.savefig(prob_name+"_Map_"+"Runtime.png")


plt.figure()
plt.plot(mapsize,score_array)
plt.xlabel('Map Size')
plt.ylabel('Average Rewards')
plt.title('Policy Iteration, Rewards')
plt.grid()
plt.savefig(prob_name+"_Map_"+"Rewards.png")


plt.figure()
plt.plot(mapsize,iter_array)
plt.xlabel('Map Size')
plt.ylabel('Iterations to Converge')
plt.title('Policy Iteration, Iterations')
plt.grid()
plt.savefig(prob_name+"_Map_"+"Iterations.png")

#Test different gamma values for PI
map_size = 8
my_map = frozen_lake.generate_random_map(size=map_size, p=0.9)
env = gym.make("FrozenLake-v1", desc=my_map)
env = env.unwrapped

time_array=[]
gamma_array=[0.15, 0.25, 0.55, 0.7, 0.8, 0.95, 0.99]
iter_array=[]
score_array=[]
for gamma in gamma_array:
    start=time.time()
    best_policy,n_iter = policy_iteration(env, gamma)
    scores = evaluate_policy(env, best_policy, gamma)
    end=time.time()
    score_array.append(np.mean(scores))
    iter_array.append(n_iter)
    time_array.append(end-start)
print(gamma_array)
print(time_array)
print(score_array)
print(iter_array)
plt.figure()
plt.plot(gamma_array,time_array)
plt.xlabel('Gammas')
plt.ylabel('Runtime')
plt.title('Policy Iteration, Runtime')
plt.grid()
plt.savefig(prob_name+"_PI_"+"G_Runtime.png")

plt.figure()
plt.plot(gamma_array,score_array)
plt.xlabel('Gammas')
plt.ylabel('Average Rewards')
plt.title('Policy Iteration, Reward Analysis')
plt.grid()
plt.savefig(prob_name+"_PI_"+"G_Rewards.png")

plt.figure()
plt.plot(gamma_array,iter_array)
plt.xlabel('Gammas')
plt.ylabel('Iterations to Converge')
plt.title('Policy Iteration, Iterations')
plt.grid()
plt.savefig(prob_name+"_PI_"+"G_Iterations.png")
#Test different gamma values for VI
map_size = 8
my_map = frozen_lake.generate_random_map(size=map_size, p=0.9)
env = gym.make("FrozenLake-v1", desc=my_map)
env = env.unwrapped

time_array=[]
gamma_array=[0.15, 0.25, 0.55, 0.7, 0.8, 0.95, 0.99]
iter_array=[]
score_array=[]
for gamma in gamma_array:
    start=time.time()
    v,n_iter = value_iteration(env, gamma)
    policy = extract_policy(env, v, gamma)
    scores = evaluate_policy(env, policy, gamma)
    end=time.time()
    score_array.append(np.mean(scores))
    iter_array.append(n_iter)
    time_array.append(end-start)
print(gamma_array)
print(time_array)
print(score_array)
print(iter_array)
plt.figure()
plt.plot(gamma_array,time_array)
plt.xlabel('Gammas')
plt.ylabel('Runtime')
plt.title('Value Iteration, Runtime')
plt.grid()
plt.savefig(prob_name+"_VI_"+"G_Runtime.png")

plt.figure()
plt.plot(gamma_array,score_array)
plt.xlabel('Gammas')
plt.ylabel('Average Rewards')
plt.title('Value Iteration, Reward Analysis')
plt.grid()
plt.savefig(prob_name+"_VI_"+"G_Rewards.png")

plt.figure()
plt.plot(gamma_array,iter_array)
plt.xlabel('Gammas')
plt.ylabel('Iterations to Converge')
plt.title('Value Iteration, Iterations')
plt.grid()
plt.savefig(prob_name+"_VI_"+"G_Iterations.png")

#check policy equal
map_size = 8
my_map = frozen_lake.generate_random_map(size=map_size, p=0.9)
env = gym.make("FrozenLake-v1", desc=my_map)
env = env.unwrapped
gamma_array=[0.15, 0.25, 0.55, 0.7, 0.8, 0.95, 0.99]
policy_equal=[]
for gamma in gamma_array:
    v,n_iter = value_iteration(env, gamma)
    policy_v = extract_policy(env, v, gamma)
    policy_p,n_iter = policy_iteration(env, gamma)
    policy_equal.append(policy_v==policy_p)
print(policy_equal)

#Q-learning for Frozen Lake
time_array=[]
gamma_array=[0.15, 0.25, 0.55, 0.7, 0.8, 0.95, 0.99]
score_array=[]
test_score_array=[]
policy = []

for gamma in gamma_array:
    start = time.time()
    agent = Agent(env, map_size,gamma)
    print("###### LEARNING #####")
    reward_total = 0.0
    for i in range(LEARNING_COUNT):
        reward = agent.learn()
        reward_total += reward
    print("gamma         : {}".format(gamma))
    print("episodes      : {}".format(LEARNING_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / LEARNING_COUNT))

    end = time.time()
    score_array.append(reward_total / LEARNING_COUNT)
    time_array.append(end-start)
    
    print("###### TEST #####")
    reward_total = 0.0
    for i in range(TEST_COUNT):
        reward = agent.test()
        reward_total += reward
    print("episodes      : {}".format(TEST_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / TEST_COUNT))
    
    policy_curr = [np.argmax(agent.q_val[state]) for state in range(map_size*map_size)]
    policy_curr = np.array(policy_curr)
    policy.append(policy_curr)
    test_score_array.append(reward_total / TEST_COUNT)

print(gamma_array)
print(time_array)
print(score_array)
plt.figure()
plt.plot(gamma_array,time_array)
plt.xlabel('Gammas')
plt.ylabel('Runtime')
plt.title('Q-Learning, Runtime')
plt.grid()
plt.savefig("FL_Runtime.png")

plt.figure()
plt.plot(gamma_array,score_array)
plt.xlabel('Gammas')
plt.ylabel('Average Rewards')
plt.title('Q-Learning, Rewards')
plt.grid()
plt.savefig("FL_Rewards.png")

plt.figure()
plt.plot(gamma_array,test_score_array)
plt.xlabel('Gammas')
plt.ylabel('Average Rewards')
plt.title('Q-Learning, Rewards(Test Cases)')
plt.grid()
plt.savefig("FL_TestRewards.png")
