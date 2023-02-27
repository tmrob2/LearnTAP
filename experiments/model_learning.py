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
from model_based.tap_models.stapu_env import STAPU
import pandas as pd
import seaborn as sns
from model_based.tap_models.cent_stapu_model import CentModel
from model_based.tap_models.dec_stapu_model import DecModel


parser = argparse.ArgumentParser(
    prog="Model Learning Experiment",
    description="Comparing ways of learning a STAPU model",
)

parser.add_argument("-l", "--load", dest="load_model", action="store_true", help="load RL agent models")
parser.add_argument("-s", "--save", dest="save_model", action="store_true", help="save RL agent models")
parser.add_argument("-d", "--data", dest="load_data", action="store_true", help="load saved dataset")
parser.add_argument("--sd", dest="save_data", action="store_true", help="save dataset")
parser.add_argument("-c", dest="cent", action="store_true", help="learn centralised model")
parser.add_argument("--dec", dest="dec", action="store_true", help="learn decentralised model")

args = parser.parse_args()

def treasure_lt_12(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if 0.1 < treasure_value <= 12. and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q

def treasure_b12_20(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if 0.1 < treasure_value <= 12. and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q

def treasure_lt_14(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if 0.1 < treasure_value <= 14. and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q

def treasure_gt_20(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value > 20. and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q

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

def num_tasks2_t1():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_lt_14)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks2_t2():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_gt_20)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks3_t1():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_lt_12)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks3_t2():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_b12_20)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks3_t3():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_gt_20)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

tasks1 = [num_tasks2_t1()]
tasks2 = [num_tasks2_t1(), num_tasks2_t2()]
tasks3 = [num_tasks3_t1(), num_tasks3_t2(), num_tasks3_t3()]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cent:
        env_shape = (11, 11)
        state_shape = env_shape
        state_space = np.prod(np.array(state_shape))
        num_trajectories = 10000
        env = DeepSeaTreasure(render_mode=None, float_state=False)
        stapu = CentModel(env, device=device, tasks=tasks3, env_shape=env_shape)
        state, _ = stapu.env.reset()
        state_size = len(state)
        if args.load_model:
            transition_checkpoint = f"/home/thomas/ai_projects/LearnTAP-v2/models/cent-transitions"
            model = TransitionModel(state_size, 64, state_space, transition_checkpoint).to(device)
            model.load_model()
            stapu.model = model 
        data = stapu.make_models(num_trajectories, batch_size=None, load=args.load_data, save=args.save_data)
            # Test the model outputs to see how well the models learned the environments
        if args.save_model:
            stapu.save_model()
            

        df = pd.DataFrame(data=data, columns=["model-type", "episode", "loss"])
        df.to_csv(f"/home/thomas/ai_projects/LearnTAP-v2/data/cent-loss-T{len(tasks3)}.csv", header=True, index=False)
    #plt.figure(figsize=(10, 5))
    #ax = sns.lineplot(data=df, x="episode", y="loss", hue="model-type", linewidth=2.5)
    #ax.set(xlabel="Epsiodes", ylabel="Loss")
    #plt.yscale('log')
    #plt.xscale('log')
    #ax.legend(title="Alg. Type", title_fontsize=13)
    #plt.show()

        stapu.test_models(100, print_states=False)
        del env, stapu, df
    

    if args.dec:

        state_shape = env_shape
        state_space = np.prod(np.array(state_shape))
        num_trajectories = 5000
        env = DeepSeaTreasure(render_mode=None, float_state=True)
        state, _ = env.reset()
        state_size = len(state) + 1
        env_ = DeepSeaTreasure(render_mode=None, float_state=False)
        mamodel = STAPU(env_, tasks3, env_shape=env_shape, float_state=False)
        dec_stapu = DecModel(env, device=device, num_actions=mamodel.action_space.n, tasks=tasks3)
        models = []
        if args.load_model:
            for a in range(env.num_agents):
                transition_checkpoint = f"/home/thomas/ai_projects/LearnTAP-v2/models/transitions_agent{a}"
                model = TransitionModel(state_size, 32, state_space, transition_checkpoint).to(device)
                model.load_model()
                models.append(model)
                dec_stapu.models = models
        dec_data = dec_stapu.make_models(num_trajectories, batch_size=None, load=args.load_data, save=False)
            # Test the model outputs to see how well the models learned the environments
        if args.save_model:
            for a in range(env.num_agents):
                dec_stapu.models[a].save_model()
            
        dec_stapu.test_models(10, print_states=False)

        df = pd.DataFrame(data=dec_data, columns=["model-type", "episode", "loss"])
        df.to_csv(f"/home/thomas/ai_projects/LearnTAP-v2/data/dec-loss-T{len(tasks3)}.csv", header=True, index=False)

    if args.cent and args.dec:
        df_full = data.extend(dec_data)
        df_full = pd.DataFrame(data=dec_data, columns=["model-type", "episode", "loss"])
        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(data=df, x="episode", y="loss", hue="model-type", linewidth=2.5)
        ax.set(xlabel="Epsiodes", ylabel="Loss")
        plt.yscale('log')
        plt.xscale('log')
        ax.legend(title="Alg. Type", title_fontsize=13)
        plt.show()