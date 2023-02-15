import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import gymnasium as gym
from mo_gym.deep_sea_treasure.deep_sea_treasure import DEFAULT_MAP, DeepSeaTreasure
import tqdm
from copy import deepcopy
import random
from more_itertools import chunked
from torch.utils.data import Dataset
from typing import Tuple, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransitionModel(nn.Module):
    def __init__(self, inputs, hidden, states, checkpoint) -> None:
        super(TransitionModel, self).__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, states)
        self.state_size = inputs
        self.checkpoint_file = checkpoint

    def forward(self, x: np.array):
        # concatenate x and a into a tensor
        x = self.fc1(torch.tanh(x))
        x = self.fc2(torch.tanh(x))
        x = self.fc3(torch.tanh(x))
        x = self.fc4(x)
        return x

    def save_model(self):
        print('... saving models ...')
        torch.save(deepcopy(self.state_dict()), self.checkpoint_file)

    def load_model(self):
        print('... loading models ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class TransitionDataset(Dataset):
    def __init__(self, X: np.array, y: np.array, env_shape: Tuple) -> None:
        y = np.ravel_multi_index(
            np.transpose((y * (np.array(env_shape) - 1))).astype(np.int32), tuple(env_shape)
        )
        self.data = X.astype(np.float32)
        self.labels = y.astype(np.float32)
        # convert the label set into classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# TODO make a non-decentralised model data collection process

def collect_batch(n_trajectories, env: gym.Env, state_shape):
    obs_dataset = []
    #xlist = []
    #ylist = []
    target_dataset = []
    #reward_target = []
    zero_base_state_shape = np.array(state_shape) - 1
    with tqdm.trange(n_trajectories) as t:
        for k in t:
            terminated, truncated = False, False
            done = terminated or truncated
            state, _ = env.reset()
            samples = 0
            while not done:
                # pick a random action from the environment
                action = env.action_space.sample()
                state = state / zero_base_state_shape
                obs_dataset.append(np.append(state, float(action / env.action_space.n)))
                obs, _, terminated, truncated, _ = env.step(action)
                # add the data to the dataset
                #zero_base_state_shape = np.array(state_shape) - 1
                #data = (obs * zero_base_state_shape).astype(np.int32)
                #xlist.append(int(obs[0] * 10.))
                #ylist.append(int(obs[1] * 10.))
                target_dataset.append(obs)
                #reward_target.append(reward) # / reward_norm)
                state = obs
                done = terminated or truncated
                samples += 1
            t.set_description(f"traj {k}")
            t.set_postfix(samples=samples)
    
    return obs_dataset, target_dataset

def obs_convert_to_torch(obs_data, obs_target, env_shape):
    tobs_data = torch.from_numpy(np.array(obs_data, dtype=np.float32)).to(device)
    t_obs_target = torch.from_numpy(np.ravel_multi_index(np.transpose(np.array(obs_target)), tuple(env_shape))).to(device)
    return tobs_data, t_obs_target

def reward_convert_to_torch(obs_data, reward_target):
    tobs_data = torch.from_numpy(np.array(obs_data, dtype=np.float32)).to(device)
    t_reward_target = torch.from_numpy(np.array(reward_target, dtype=np.float32)).to(device)
    return tobs_data, t_reward_target

def batch_data(iterable, n=1, shuffle=False):
    if shuffle:
        random.shuffle(iterable)
    return list(chunked(iterable, n))
    

def train(batch, target, model, optimiser, criterion):
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

def main(n_trajectories, state_space, transition_checkpoint, rewards_checkpoint):
    # initialise the DST environment
    env = DeepSeaTreasure(render_mode=None, dst_map=DEFAULT_MAP, float_state=True)
    state, _ = env.reset()
    state_size = len(state) + 1
    state_shape = (11, 11)
    model = TransitionModel(state_size, 32, state_space, transition_checkpoint).to(device)
    #rewards_model = RewardModel(state_size, 16, env.reward_space.high.shape[0], rewards_checkpoint).to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    reward_optimiser = optim.Adam(rewards_model.parameters(), lr=0.001)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    state_action, sprime_target, reward_target = collect_batch(
            n_trajectories=n_trajectories, env=env,
            state_shape=state_shape
    )
    #loss = 1.
    #epoch = 0
    batch_size = 1000
    batched_obs, batched_target, batched_reward = \
        batch_data(state_action, batch_size), batch_data(sprime_target, batch_size), batch_data(batch_size)
    with tqdm.trange(1000) as t:
        for i in t: # loop over the data set multiple times
            # collect some trajectories from the environment
            # train the neural network
            for batch_no in len(batched_obs):
                tobs, tsprime = obs_convert_to_torch(
                    batched_obs[batch_no], batched_target[batch_no])
                loss = train(tobs, tsprime, model, optimiser, criterion1)
                t.set_description("Trans Model")
                t.set_postfix(epoch=i, batch=batch_no, loss=loss)

    return model

if __name__ == "__main__":
    S = 11 * 11
    model, rewards_model = main(50000, S)
    # test the model
    env = DeepSeaTreasure(render_mode=None, dst_map=DEFAULT_MAP, float_state=True)
    state, _ = env.reset()
    #state_shape = len(state) + 1
    env_shape = (11, 11)
    state_action, sprime, r_target = collect_batch(10, env, env_shape)
    obs, y = obs_convert_to_torch(state_action, sprime, env_shape)
    with torch.no_grad():
        pred_state = model(obs)
        pred_state = pred_state.max(1, keepdim=True)[1]
    pred = pred_state.detach().cpu().numpy().squeeze()
    target = y.detach().cpu().numpy()
    for i in range(obs.shape[0]):
        print(f"predicted state: {pred[i]}, obs state: {target[i]}")
    print("Error", float(np.count_nonzero(pred_state == target) / np.size(target)) * 100.)
