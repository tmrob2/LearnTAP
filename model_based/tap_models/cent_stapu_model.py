from mo_gym.deep_sea_treasure.deep_sea_treasure import DEFAULT_MAP
from model_based.ma_envs.multi_agent_deep_sea_treasure.ma_deep_sea_treasure import DeepSeaTreasure
import matplotlib.pyplot as plt
from utils.dfa import DFA
from model_based.learn_models.dst_nn import TransitionModel, TransitionDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from numpy import load, save
from collections import deque
import argparse
import numpy as np
import gymnasium as gym
import tqdm
from typing import Tuple, List 
from model_based.tap_models.stapu_env import STAPU
import pandas as pd
import seaborn as sns

# define a simple test task to make sure that this is working
# A DFA is defined with tranisition functions 
# Informally, let the task be find two pieces of treasure
#Qstates = [0, 1, 2] # 0 is initial, 1 is found treasure, 2 is fail
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CentModel:
    def __init__(self, env: gym.Env, device, tasks: List[DFA], env_shape):
        self.env = STAPU(env, tasks, float_state=False, env_shape=env_shape)
        self.state_shape = self.env.state_shape
        self.num_agents = env.num_agents
        self.device = device
        self.num_actions = self.env.action_space.n
        self.model = None

    def _collect_batch(self, n_trajectories):
        # Assumes a multiagent environment
        assert self.num_agents >= 2
        # Collect n_trajectories for each agent in the environment.
        # This will be used to train each agent's model of the environment
        # 
        # The batch dataset will be tranformed into a data loader
        obs_dataset = []
        target_dataset = []
        with tqdm.trange(n_trajectories) as t:
            for k in t:
                terminated = False
                state, _ = self.env.reset()
                state = self.env.convert_state_to_float(state)
                samples = 0
                while not terminated:
                    # pick a random action from the environment
                    action = self.env.action_space.sample()
                    obs_dataset.append(np.append(state, float(action / self.env.action_space.n)))
                    obs, _, terminated, _, _ = self.env.step(action)
                    obs_ = self.env.convert_state_to_float(obs)
                    target_dataset.append(obs_)
                    samples += 1
                    state = obs_
                t.set_description(f"traj {k}")
                t.set_postfix(trajectory=k)
        return obs_dataset, target_dataset

    def _make_data(self, n_trajectories, batch_size=None, save_=False, load_=False, test=False):
        #batch_sizes = []
        if not load_:
            observations, targets = self._collect_batch(n_trajectories)
        #if test:
        #    for a in range(self.num_agents):
        #        batch_sizes.append(len(observations[a]))
        if load_:
            X = load(f"/home/thomas/ai_projects/LearnTAP-v2/models/cent-obs-T{len(self.env.tasks)}.npy")
            y = load(f"/home/thomas/ai_projects/LearnTAP-v2/models/cent-target-T{len(self.env.tasks)}.npy")
        else: 
            X, y = np.array(observations), np.array(targets)
        if save_:
            save(f"/home/thomas/ai_projects/LearnTAP-v2/models/cent-obs-T{len(self.env.tasks)}.npy", X)
            save(f"/home/thomas/ai_projects/LearnTAP-v2/models/cent-target-T{len(self.env.tasks)}.npy", y)

        # construct the dataset
        dataset = TransitionDataset(X, y, self.env.state_shape)
        # construct a loader
        if batch_size is None:
            batch_size = len(X) // 200
        else:
            batch_size = len(X)
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader

    def _learn(self, batch, target, model, optimiser, criterion):
        # zero the gradient buffers of all params and backprops
        # with random gradients
        #
        # get data from nn
        obs = model(batch)
        loss = criterion(obs, target) #+ criterion(rtarget, reward)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        return loss

    def make_models(self, n_trajectories, batch_size=None, load=False, save=False):
        state, _ = self.env.reset()
        state_size = len(state) + 1
        state_shape = self.env.state_shape
        state_space = np.prod(np.array(state_shape))
        models = []
        transition_checkpoint = f"/home/thomas/ai_projects/LearnTAP-v2/models/cent-transitions"
        model = TransitionModel(state_size, 32, state_space, transition_checkpoint).to(device)
        self.model = model
        optimiser = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        loader = self._make_data(n_trajectories, batch_size, save, load)
        # train each agent model in sequence
        avg_loss = 1.
        losses = deque([], maxlen=100)
        data = []
        with tqdm.trange(1000) as t:
            for i in t: # loop over the data set multiple times
                # collect some trajectories from the environment
                # train the neural network
                for batch_idx, (batch, labels) in enumerate(loader):
                    batch, labels = batch.to(device), labels.type(torch.long).to(device)
                    loss = self._learn(batch, labels, self.model, optimiser, criterion)
                    t.set_description(f"Training Model")
                    float_loss = loss.detach().cpu().numpy()
                    printloss = f"{float_loss:.3f}"
                    data.append(["cent", i, float(float_loss)])
                    t.set_postfix(e=i, b=batch_idx, l=printloss, avg=avg_loss)
                    losses.append(loss.detach().cpu().numpy())
                    t.set_postfix(e=i, b=batch_idx, avg=avg_loss, l=printloss)
                if i > 10:
                    avg_loss = np.mean(np.array(losses))
                if avg_loss < 0.01:
                    break
        self.models = models
        return data

    def test_models(self, test_trajectories, print_states=False):
        loader = self._make_data(n_trajectories=test_trajectories, batch_size=0, test=True)
        with torch.no_grad():
            for batch, labels in loader:
                batch = batch.to(device)
                pred_state = self.model(batch)
                pred_state = pred_state.max(1, keepdim=True)[1]
                pred = pred_state.detach().cpu().numpy().squeeze()
                target = labels.cpu().numpy()
                states = batch[:, :-1] * (torch.tensor(self.env.state_shape).to(device) - 1.)
                actions = batch[:, -1] * self.env.action_space.n
                print_obs = torch.cat((states, actions.unsqueeze(1)), 1)
                print_obs = print_obs.detach().cpu().numpy()
                for i in range(batch.shape[0]):
                    predicted_full = np.unravel_index(pred[i], self.env.state_shape)
                    obs_full = np.unravel_index(target[i].astype(np.int32), self.env.state_shape)
                    if print_states:
                        print(f"{print_obs[i, :]} => predicted state: {predicted_full}, obs state: {obs_full}, correct: {pred[i] == target[i]}")
                err = 100. - (np.count_nonzero(pred == target)) / np.size(target) * 100.
                print(f"Agent model error: {err:.3f}")

    def _make_state(self, state_idx):
        return np.array(np.unravel_index(state_idx, self.state_shape), dtype=np.float32) * 0.1

    def _activate_task(self, action, task, data):
        if action == task + 6:
            data["activate"] = True
        else:
            data["activate"] = False
        return data

    def __call__(self, state, action: int, p):
        # Given a state, and action, combine and convert it to float
        # enter it into the model, get an observation
        # compute the reward of the input state, and action
        # if the reward is -10 then return p - 0.2
        
        Q = state[3:]
        R = [0.] * (self.num_agents + len(self.tasks))
        inProgress = any([t.model_in_progress(Q[j]) for j, t in enumerate(self.tasks)])
        rxint, ryint = tuple((np.array([state[1], state[2]]) * (np.array(self.env.env_shape) -1 )).astype(np.int32))
        value = self.env.get_map_value((rxint, ryint))
        state = np.array(state) / (np.array(self.env.state_shape) - 1)
        # Combine state and action
        action = float(action / self.env.action_space.n)
        input_state = np.append(state, action)
        obs = self.model(input_state)
        # convert back to the original state format
        obs_ = obs * (np.array(self.env.state_shape) - 1)
        
        # Compute the rewards
        if inProgress and action < 4:
            R[int(state[0])] = -1. if value >= 0 else value
        else:
            R[int(state[0])] = 0. if value >= 0 else value
        # get the treasure value of the active agent
        R[self.num_agents:] = [
            self.tasks[j].model_rewards(Q[j], p)
            for j in range(len(self.tasks))
        ]
        return tuple(obs_), R, p
    

