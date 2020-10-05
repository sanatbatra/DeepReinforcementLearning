from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from BehaviourCloning.model import Model
from BehaviourCloning.utils import *
from collections import deque

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    state = rgb2gray(state)
    # state = state
    state_list = [state for _ in range(history_length)]
    count = 0
    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        state_list.pop(0)

        state_list.append(state)
        # print(state_list)
        state_list_list = np.array(state_list)
        # print(state_list_list)
        state_list_list = torch.tensor(state_list_list)

        state_list_list = state_list_list.reshape(-1, history_length, 96, 96)

        
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])

        a = agent(state_list_list)
        a = a[0]

        a[1] = max(0.0, a[1])
        a[2] = max(0.0, a[2])

        a = a.detach().numpy()
        # a[1] = max(0, a[1])
        # a[2] = max(0, a[2])

        print(a)
        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        state = rgb2gray(state)
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 10                 # number of episodes to test
    history_length = 5
    # TODO: load agent
    agent = Model(history_length)
    agent.load_state_dict(torch.load('models/agent_20000.pth'))
    agent.eval()

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent_20000-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
