import numpy as np
import pandas as pd
import scipy
import time
import sys
import matplotlib.pyplot as plt
import mdptoolbox, mdptoolbox.example
prob_name = "Forest"
#test different number of states
mean_value = []
policy = []
iters = []
time_array = []
param_name = "Num Of States"
save_filename = "S"
iter_range = [10, 100, 200, 500, 1000]
for i in iter_range:
    P, R = mdptoolbox.example.forest(S=i, r1=4, r2=2, p=0.1)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    pi.run()
    mean_value.append(np.mean(pi.V))
    policy.append(pi.policy)
    iters.append(pi.iter)
    time_array.append(pi.time)
plt.figure()
plt.plot(iter_range, time_array)
plt.xlabel(param_name)
plt.ylabel('Runtime')
plt.title('Initial Runs, Runtime Curve')
plt.grid()
plt.savefig(prob_name+"_"+save_filename+"_Runtime.png")

plt.figure()
plt.plot(iter_range,mean_value)
plt.xlabel(param_name)
plt.ylabel('Average Rewards')
plt.title('Initial Runs, Reward Curve')
plt.grid()
plt.savefig(prob_name+"_"+save_filename+"_Rewards.png")

plt.figure()
plt.plot(iter_range,iters)
plt.xlabel(param_name)
plt.ylabel('Iterations')
plt.title('Initial Runs, Iterations Curve')
plt.grid()
plt.savefig(prob_name+"_"+save_filename+"_Iters.png")
#test different rewards
mean_value = []
policy = []
iters = []
time_array = []
param_name = "Wait Reward"
save_filename = "R"
iter_range = [1, 2, 4, 5, 10]
for i in iter_range:
    P, R = mdptoolbox.example.forest(S=1000, r1=i, r2=2, p=0.1)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    pi.run()
    mean_value.append(np.mean(pi.V))
    policy.append(pi.policy)
    iters.append(pi.iter)
    time_array.append(pi.time)
plt.figure()
plt.plot(iter_range, time_array)
plt.xlabel(param_name)
plt.ylabel('Runtime')
plt.title('Initial Runs, Runtime Curve')
plt.grid()
plt.savefig(prob_name+"_"+save_filename+"_Runtime.png")

plt.figure()
plt.plot(iter_range,mean_value)
plt.xlabel(param_name)
plt.ylabel('Average Rewards')
plt.title('Initial Runs, Reward Curve')
plt.grid()
plt.savefig(prob_name+"_"+save_filename+"_Rewards.png")

plt.figure()
plt.plot(iter_range,iters)
plt.xlabel(param_name)
plt.ylabel('Iterations')
plt.title('Initial Runs, Iterations Curve')
plt.grid()
plt.savefig(prob_name+"_"+save_filename+"_Iters.png")
#test different prob
mean_value = []
policy = []
iters = []
time_array = []
param_name = "Probability"
save_filename = "P"
iter_range = [0.01, 0.02, 0.05, 0.1, 0.2]
for i in iter_range:
    P, R = mdptoolbox.example.forest(S=1000, r1=4, r2=2, p=i)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    pi.run()
    mean_value.append(np.mean(pi.V))
    policy.append(pi.policy)
    iters.append(pi.iter)
    time_array.append(pi.time)
plt.figure()
plt.plot(iter_range, time_array)
plt.xlabel(param_name)
plt.ylabel('Runtime')
plt.title('Initial Runs, Runtime Curve')
plt.grid()
plt.savefig(prob_name+"_"+save_filename+"_Runtime.png")

plt.figure()
plt.plot(iter_range,mean_value)
plt.xlabel(param_name)
plt.ylabel('Average Rewards')
plt.title('Initial Runs, Reward Curve')
plt.grid()
plt.savefig(prob_name+"_"+save_filename+"_Rewards.png")

plt.figure()
plt.plot(iter_range,iters)
plt.xlabel(param_name)
plt.ylabel('Iterations')
plt.title('Initial Runs, Iterations Curve')
plt.grid()
plt.savefig(prob_name+"_"+save_filename+"_Iters.png")
#PI different gamma
P, R = mdptoolbox.example.forest(S=1000, r1=4, r2=2, p=0.1)
mean_value = []
policy = []
iters = []
time_array = []
gamma_array = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
for i in gamma_array:
    pi = mdptoolbox.mdp.PolicyIteration(P, R, i)
    pi.run()
    mean_value.append(np.mean(pi.V))
    policy.append(pi.policy)
    iters.append(pi.iter)
    time_array.append(pi.time)
plt.figure()
plt.plot(gamma_array, time_array)
plt.xlabel('Gamma')
plt.title('Policy Iteration, Runtime')
plt.ylabel('Execution Time (s)')
plt.grid()
plt.savefig(prob_name+"_PI_Runtime.png")

plt.figure()
plt.plot(gamma_array,mean_value)
plt.xlabel('Gamma')
plt.ylabel('Average Rewards')
plt.title('Policy Iteration, Rewards')
plt.grid()
plt.savefig(prob_name+"_PI_Rewards.png")

plt.figure()
plt.plot(gamma_array,iters)
plt.xlabel('Gamma')
plt.ylabel('Iterations')
plt.title('Policy Iteration, Iterations')
plt.grid()
plt.savefig(prob_name+"_PI_Iterations.png")
#VI different gamma
P, R = mdptoolbox.example.forest(S=1000, r1=4, r2=2, p=0.1)
mean_value = []
policy = []
iters = []
time_array = []
gamma_array = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
for i in gamma_array:
    pi = mdptoolbox.mdp.ValueIteration(P, R, i)
    pi.run()
    mean_value.append(np.mean(pi.V))
    policy.append(pi.policy)
    iters.append(pi.iter)
    time_array.append(pi.time)
plt.figure()
plt.plot(gamma_array, time_array)
plt.xlabel('Gamma')
plt.title('Value Iteration, Runtime')
plt.ylabel('Execution Time (s)')
plt.grid()
plt.savefig(prob_name+"_VI_Runtime.png")

plt.figure()
plt.plot(gamma_array,mean_value)
plt.xlabel('Gamma')
plt.ylabel('Average Rewards')
plt.title('Value Iteration, Rewards')
plt.grid()
plt.savefig(prob_name+"_VI_Rewards.png")

plt.figure()
plt.plot(gamma_array,iters)
plt.xlabel('Gamma')
plt.ylabel('Iterations')
plt.title('Value Iteration, Iterations')
plt.grid()
plt.savefig(prob_name+"_VI_Iterations.png")

#Compare Policy - PI vs VI
P, R = mdptoolbox.example.forest(S=1000, r1=4, r2=2, p=0.1)
gamma_array = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
equal_array = []
for i in gamma_array:
    pi1 = mdptoolbox.mdp.ValueIteration(P, R, i)
    pi1.run()
    pi2 = mdptoolbox.mdp.PolicyIteration(P, R, i)
    pi2.run()
    equal_array.append(pi1.policy==pi2.policy)
print(equal_array)

#different gamma for Q-learning
P, R = mdptoolbox.example.forest(S=1000, r1=4, r2=2, p=0.1)
mean_value = []
time_array = []
gamma_array = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
for i in gamma_array:
    start = time.time()
    pi = mdptoolbox.mdp.QLearning(P,R,i,n_iter=1000000)
    run_stats = pi.run()
    end = time.time()
    mean_value.append(np.mean(pi.V))
    time_array.append(end-start)
plt.figure()
plt.plot(gamma_array,mean_value)
plt.xlabel('Gamma')
plt.ylabel('Average Rewards')
plt.title('Q-Learning, Rewards')
plt.grid()
plt.savefig(prob_name+"_Q_Rewards.png")
plt.figure()
plt.plot(gamma_array, time_array)
plt.xlabel('Gamma')
plt.ylabel('Runtime')
plt.title('Q-Learning, Runtime')
plt.grid()
plt.savefig(prob_name+"_Q_Runtime.png")
