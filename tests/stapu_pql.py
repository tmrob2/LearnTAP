from mo_gym.deep_sea_treasure.deep_sea_treasure import DEFAULT_MAP
import numpy as np
from model_based.mo.agents.mamo_dyna_pql import DynaPQL
from model_based.ma_envs.multi_agent_deep_sea_treasure.ma_deep_sea_treasure import DeepSeaTreasure
from model_based.tap_models.stapu_env import STAPU
import matplotlib.pyplot as plt
from utils.dfa import DFA
import torch
from tests.dec_stapu_model_pql import DecModel
from model_based.learn_models.dst_nn import TransitionModel
import argparse


# define a simple test task to make sure that this is working
# A DFA is defined with tranisition functions 
# Informally, let the task be find two pieces of treasure
#Qstates = [0, 1, 2] # 0 is initial, 1 is found treasure, 2 is fail
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    prog="LearnTAP",
    description="STAPU model based reainforcement learning",
)

parser.add_argument("-l", "--load", dest="load_model", action="store_true", help="load RL agent models")
parser.add_argument("-s", "--save", dest="save_model", action="store_true", help="save RL agent models")
parser.add_argument("-d", "--data", dest="load_data", action="store_true", help="load saved dataset")

args = parser.parse_args()


def treasure1_found(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if 0.1 < treasure_value < 10. and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q

def treasure2_found(q, data, agent):
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

def find_treasure1():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure1_found)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def find_treasure2():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure2_found)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

tasks = [find_treasure1(), find_treasure2()]
    
def test_model_based_pql():
    env_shape = (11, 11)
    state_shape = env_shape
    state_space = np.prod(np.array(state_shape))
    num_trajectories = 5000
    env = DeepSeaTreasure(render_mode=None, float_state=True)
    state, _ = env.reset()
    state_size = len(state) + 1
    env_ = DeepSeaTreasure(render_mode=None, float_state=False)
    mamodel = STAPU(env_, tasks, env_shape=env_shape, float_state=False)
    dec_stapu = DecModel(env, device=device, num_actions=mamodel.action_space.n, tasks=tasks)
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
    #ref_point = np.array([0] * env.num_agents + [-25] * len(tasks))
    ref_point = np.array([-100, 0, 0])
    num_episodes = 100
    agent = DynaPQL(
        mamodel,
        ref_point,
        gamma=1.0,
        initial_epsilon=1.,
        epsilon_decay=0.9,
        final_epsilon=0.1,
        seed=1,
        log=True,
        planning=True,
        model=dec_stapu,
        tasks=tasks,
        mcts=True
    )
    # first learn the STAPU model
    #stapu_state_space = np.prod(mamodel.state_shape)
    #Training
    pf = agent.train(
        num_episodes=num_episodes, log_every=1, 
        action_eval="hypervolume", k=50, max_steps=30
    )
    #assert len(pf) > 0
    #print(pf)
    #x, y, z = zip(*list(pf))
    ## put this one on the first sub-figure
    #plt.scatter(x, z)
    ## put this one on the second sub-figure
    #plt.scatter(y, z)
    #plt.show()
    ## Policy following
    #target = np.array(pf.pop())
    #print(f"Tracking {target}")
    ## Fix tracking with STAPU environment
    ##reward = agent.track_policy(target)
    #print(f"Obtained {reward}")

test_model_based_pql()
