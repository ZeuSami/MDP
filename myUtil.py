"""
@Source 1
**************************************************************************************
*    Title: Deep Reinforcement Learning Demysitifed Policy Iteration, Value Iteration and Q-learning
*    Author: Moustafa Alzantot
*    Date: 2017
*    Code version: N/A
*    Availability: https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
*
***************************************************************************************
@Source 2
**************************************************************************************
*    Title: FrozenLake-v0 with Q learning
*    Author: jojonki
*    Date: 2021
*    Code version: N/A
*    Availability: https://gist.github.com/jojonki/6291f8c3b19799bc2f6d5279232553d7
*
***************************************************************************************
"""
import numpy as np
import gym
from gym import wrappers
import time
##https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
def run_episode(env, policy, gamma, render = True):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    start = time.time()
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

def extract_policy(env,v, gamma):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.desc.shape[0]*env.desc.shape[1])
    for s in range(env.desc.shape[0]*env.desc.shape[1]):
        q_sa = np.zeros(4)
        for a in range(4):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma):
    v = np.zeros(env.desc.shape[0]*env.desc.shape[1])
    eps = 1e-5
    while True:
        prev_v = np.copy(v)
        for s in range(env.desc.shape[0]*env.desc.shape[1]):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v

def evaluate_policy(env, policy, gamma , n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def policy_iteration(env, gamma):
    policy = np.random.choice(4, size=(env.desc.shape[0]*env.desc.shape[1]))  
    max_iters = 3000
    for i in range(max_iters):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env,old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            n_iter=i+1
            break
        policy = new_policy
        n_iter = i+1
    return policy,n_iter

def value_iteration(env, gamma):
    max_iters = 10000 #default max iterations
    eps = 1e-20 #default stopping criteria
    v = np.zeros(env.desc.shape[0]*env.desc.shape[1])
    for i in range(max_iters):
        prev_v = np.copy(v)
        for s in range(env.desc.shape[0]*env.desc.shape[1]):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(4)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            n_iter = i+1
            break
        n_iter = i+1
    return v, n_iter

# Q learning params
ALPHA = 0.1 # learning rate
LEARNING_COUNT = 10000
TEST_COUNT = 1000
TURN_LIMIT = 50000

class Agent:
    def __init__(self, env, map_size, gamma):
        self.env = env
        self.episode_reward = 0.0
        self.epsilon = 1.0
        self.q_val = np.zeros(map_size * map_size * 4).reshape(map_size*map_size, 4).astype(np.float32)
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.gamma = gamma

    def learn(self):
        #one episode learning
        state = self.env.reset()
        total_reward = 0.0
        
        for t in range(TURN_LIMIT):
            pn = np.random.random()
            if pn < self.epsilon:
                act = self.env.action_space.sample()
            else:
                act = self.q_val[state].argmax()
            next_state, reward, done, info = self.env.step(act)
            total_reward += reward
            q_next_max = np.max(self.q_val[next_state])
            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')
            self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act] + ALPHA * (reward + self.gamma * q_next_max - self.q_val[state, act])
            if done or t == TURN_LIMIT - 1:
                return total_reward
            else:
                state = next_state
        self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon)

    def test(self):
        state = self.env.reset()
        total_reward = 0.0
        for t in range(TURN_LIMIT):
            act = np.argmax(self.q_val[state])
            next_state, reward, done, info = self.env.step(act)
            total_reward += reward
            if done or t == TURN_LIMIT - 1:
                return total_reward
            else:
                state = next_state
        return 0.0 # over limit
