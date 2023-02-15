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
from model_based.mo.agents.failed_projects.model_based_pql import DynaPQL
from model_based.tap_models.stapu_env import STAPU

parser = argparse.ArgumentParser(
    prog="LearnTAP",
    description="STAPU model based reainforcement learning",
)

parser.add_argument("-l", "--load", dest="load_model", action="store_true", help="load RL agent models")
parser.add_argument("-s", "--save", dest="save_model", action="store_true", help="save RL agent models")
parser.add_argument("-d", "--data", dest="load_data", action="store_true", help="load saved dataset")

args = parser.parse_args()

# define a simple test task to make sure that this is working
# A DFA is defined with tranisition functions 
# Informally, let the task be find two pieces of treasure
Qstates = [0, 1, 2] # 0 is initial, 1 is found treasure, 2 is fail
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecModel:
    def __init__(self, env: gym.Env, device, num_actions, tasks: List[DFA]):
        self.env = env
        self.state_shape = env.env_shape
        self.num_agents = env.num_agents
        self.device = device
        self.num_actions = num_actions
        self.models = []
        self.tasks: List[DFA] = tasks

    def _collect_batch(self, n_trajectories):
        # Assumes a multiagent environment
        assert self.num_agents >= 2
        # Collect n_trajectories for each agent in the environment.
        # This will be used to train each agent's model of the environment
        # 
        # The batch dataset will be tranformed into a data loader
        obs_dataset = {a: [] for a in range(self.num_agents)}
        target_dataset = {a: [] for a in range(self.num_agents)}
        with tqdm.trange(n_trajectories) as t:
            for k in t:
                terminated = [False] * self.env.num_agents
                terminated_ = terminated[:]
                states, _ = self.env.reset()
                samples = 0
                while any(not x for x in terminated):
                    # pick a random action from the environment
                    actions = []
                    for a in range(self.num_agents):
                        if not terminated_[a]:
                            action = self.env.action_space.sample()
                            actions.append(action)
                            obs_dataset[a].append(np.append(states[a], float(action / self.env.action_space.n)))
                        else:
                            actions.append(-1)
                    obs, _, terminated, _, _ = self.env.step(actions)
                    for a in range(self.env.num_agents): 
                        if not terminated_[a]:
                            target_dataset[a].append(obs[a])
                    terminated_ = terminated[:]
                    states = obs
                    samples += 1
                t.set_description(f"traj {k}")
                t.set_postfix(agent=a, samples=samples)
        return obs_dataset, target_dataset

    def _make_data(self, n_trajectories, batch_size=None, save_=False, load_=False, test=False):
        batch_sizes = []
        if not load_:
            observations, targets = self._collect_batch(n_trajectories)
        if test:
            for a in range(self.num_agents):
                batch_sizes.append(len(observations[a]))
        loaders = []
        for a in range(self.num_agents):
            if load_:
                X = load(f"/home/thomas/ai_projects/LearnTAP-v2/models/obs_agent{a}.npy")
                y = load(f"/home/thomas/ai_projects/LearnTAP-v2/models/target_agent{a}.npy")
            else: 
                X, y = np.array(observations[a]), np.array(targets[a])
            if save_:
                save(f"/home/thomas/ai_projects/LearnTAP-v2/models/obs_agent{a}.npy", X)
                save(f"/home/thomas/ai_projects/LearnTAP-v2/models/target_agent{a}.npy", y)

            # construct the dataset
            dataset = TransitionDataset(X, y, self.env.env_shape)
            # construct a loader
            if batch_size is None:
                batch_size = len(X) // 5
            loader = DataLoader(dataset, 
                batch_size=batch_size if not test else batch_sizes[a])
            loaders.append(loader)
        return loaders

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
        state_shape = self.env.env_shape
        state_space = np.prod(np.array(state_shape))
        models = []
        for a in range(self.num_agents):
            transition_checkpoint = f"/home/thomas/ai_projects/LearnTAP-v2/models/transitions_agent{a}"
            model = TransitionModel(state_size, 32, state_space, transition_checkpoint).to(device)
            models.append(model)
        optimisers = [optim.Adam(models[a].parameters(), lr=0.01) for a in range(self.num_agents)]
        criterion = nn.CrossEntropyLoss()
        loaders = self._make_data(n_trajectories, batch_size, save, load)
        # train each agent model in sequence
        for a in range(self.num_agents):
            avg_loss = 1.
            losses = deque([], maxlen=100)
            with tqdm.trange(1000) as t:
                for i in t: # loop over the data set multiple times
                    # collect some trajectories from the environment
                    # train the neural network
                    for batch_idx, (batch, labels) in enumerate(loaders[a]):
                        batch, labels = batch.to(device), labels.type(torch.long).to(device)
                        loss = self._learn(batch, labels, models[a], optimisers[a], criterion)
                        t.set_description(f"Trans Model: {a}")
                        printloss = f"{loss.detach().cpu().numpy():.3f}"
                        t.set_postfix(e=i, b=batch_idx, l=printloss, avg=avg_loss)
                        losses.append(loss.detach().cpu().numpy())
                        t.set_postfix(a=a, e=i, b=batch_idx, avg=avg_loss, l=printloss)
                    if i > 10:
                        avg_loss = np.mean(np.array(losses))
                    if avg_loss < 0.01:
                        break
        self.models = models

    def test_models(self, test_trajectories, print_states=False):
        loaders = self._make_data(n_trajectories=test_trajectories, batch_size=0, test=True)
        for a in range(self.num_agents):
            with torch.no_grad():
                for batch, labels in loaders[a]:
                    batch = batch.to(device)
                    pred_state = self.models[a](batch)
                    pred_state = pred_state.max(1, keepdim=True)[1]
                    pred = pred_state.detach().cpu().numpy().squeeze()
                    target = labels.cpu().numpy()
                    states = batch[:, :-1] * (torch.tensor(self.env.env_shape).to(device) - 1.)
                    actions = batch[:, -1] * self.env.action_space.n
                    print_obs = torch.cat((states, actions.unsqueeze(1)), 1)
                    print_obs = print_obs.detach().cpu().numpy()
                    for i in range(batch.shape[0]):
                        predicted_full = np.unravel_index(pred[i], self.env.env_shape)
                        obs_full = np.unravel_index(target[i].astype(np.int32), self.env.env_shape)
                        if print_states:
                            print(f"{print_obs[i, :]} => predicted state: {predicted_full}, obs state: {obs_full}, correct: {pred[i] == target[i]}")
                    err = 100. - (np.count_nonzero(pred == target)) / np.size(target) * 100.
                    print(f"Agent {a} model error: {err:.3f}")

    def _make_state(self, state_idx):
        return np.array(np.unravel_index(state_idx, self.state_shape), dtype=np.float32) * 0.1

    def _activate_task(self, action, task, data):
        if action == task + 6:
            data["activate"] = True
        else:
            data["activate"] = False
        return data

    def __call__(self, state, action: int, p):
        # we want to call each of the agent models
        
        # first decompose the state, action pair
        #print("model input", state)
        active_agent, x, y, Q = state[0], state[1], state[2], list(state)[3:]
        rxint, ryint = tuple((np.array([x, y]) * (np.array(self.env.env_shape) -1 )).astype(np.int32))
        value = self.env.get_map_value((rxint, ryint))
        # If the action is [0, 4) then perform an agent 
        # action in the environment
        R = [0.] * (self.num_agents + len(self.tasks))
        inProgress = any([t.model_in_progress(Q[j]) for j, t in enumerate(self.tasks)])
        if action < 4 and inProgress:
            # assume that x, y are already in float fmt
            #model_input = torch.tensor([x, y, float(action / 4)], dtype=torch.float32).to(self.device)
            # which model?
            #with torch.no_grad():
            #    class_predictions = self.models[int(active_agent)](model_input)
            #    pred = class_predictions.argmax()
            #pred = pred.detach().cpu().numpy().squeeze()
            #obs = self._make_state(int(pred))
            #x, y = obs[0], obs[1]
            #next_state, _, _, _, _ = self.env()
            next_state = np.array([x, y]) * 10 + self.env.dir[action]
            if self.env.is_valid_state(np.round(next_state, 1)):
                x, y = next_state[0] * 0.1, next_state[1] * 0.1
        elif action == 4 and active_agent < self.num_agents - 1:
            # Switch transition
            active_agent += 1
            # we know that the agent is going to be in its initial position.
            #self.env.agents[int(active_agent)].reset()
            x, y = self.env.agent_init_pos[int(active_agent)]
        xint, yint = tuple((np.array([x, y]) * (np.array(self.env.env_shape) -1 )).astype(np.int32))
        if self.env.get_map_value((xint, yint)) < 0:
            p -= 0.2

        data = {"env": self.env, "x": xint, "y": yint, "activate": False, "update": False, "p": p}
        Qprime = [float(self.tasks[j].next(
            int(Q[j]), 
            self._activate_task(action, j, data), 
            active_agent))
            for j in range(len(self.tasks))
        ]
        
        if inProgress and action < 4:
            R[int(active_agent)] = -1. if value >= 0 else value
        else:
            R[int(active_agent)] = 0. if value >= 0 else value
        # get the treasure value of the active agent
        R[self.num_agents:] = [
            self.tasks[j].model_rewards(Q[j], p)
            for j in range(len(self.tasks))
        ]
        stapu_state = [active_agent, x, y]
        stapu_state.extend(Qprime)
        #print("stapu state", stapu_state)
        return tuple(stapu_state), R, p

def treasure_found(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    treasure_value = env.get_map_value((x, y))
    if treasure_value == 0:
        return q
    elif treasure_value in [11.5]:
        return 2
    else:
        return 4

def just_finished(q, data, agent):
    return 3

# Define the sink states
def accepting(q, data, agent):
    return 3

def failure(q, data, agent):
    return 4

def activate(q, data, agent):
    if data["activate"] and q == 0:
        return 1
    else:
        return 0

def find_a_treasure():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_found)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

tasks = [find_a_treasure()]

if __name__ == "__main__": 
    env_shape = (11, 11)
    env = DeepSeaTreasure(float_state=True, num_agents=2)
    state, _ = env.reset()
    state_size = len(state) + 1
    state_shape = env.env_shape
    state_space = np.prod(np.array(state_shape))
    num_trajectories = 5000
    mamodel = STAPU(env, tasks, env_shape=env_shape, float_state=False)
    dec_stapu = DecModel(env, device=device, num_actions=mamodel.action_space.n, tasks=tasks)
    ref_point = np.array([-25, 0])
    models = []
    if args.load_model:
        for a in range(env.num_agents):
            transition_checkpoint = f"/home/thomas/ai_projects/LearnTAP-v2/models/transitions_agent{a}"
            model = TransitionModel(state_size, 32, state_space, transition_checkpoint).to(device)
            model.load_model()
            models.append(model)
            dec_stapu.models = models
    else:
        dec_stapu.make_models(num_trajectories, batch_size=None, load=args.load_data, save=False)
        # Test the model outputs to see how well the models learned the environments
    if args.save_model:
        for a in range(env.num_agents):
            dec_stapu.models[a].save_model()

    dec_stapu.test_models(10, print_states=False)

    agent = DynaPQL(
        mamodel,
        dec_stapu,
        ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay=0.997,
        final_epsilon=0.2,
        seed=1,
        log=True,
        tasks=tasks
    )

    pf = agent.train(500, log_every=1, action_eval="hypervolume")

