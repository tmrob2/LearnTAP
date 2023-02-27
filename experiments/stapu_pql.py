from mo_gym.deep_sea_treasure.deep_sea_treasure import DEFAULT_MAP
import numpy as np
from model_based.mo.agents.mamo_mb_pql import DynaPQL
from model_based.ma_envs.multi_agent_deep_sea_treasure.ma_deep_sea_treasure import DeepSeaTreasure
from model_based.tap_models.stapu_env import STAPU
import matplotlib.pyplot as plt
from utils.dfa import DFA
import torch
from model_based.tap_models.dec_stapu_model import DecModel
from model_based.learn_models.dst_nn import TransitionModel
import argparse
import pandas as pd


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
parser.add_argument("--learning", dest="learning", action="store_true", help="learn a model of the environment")

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
    
def treasure_1(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 0.7 and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q
    
def treasure_2(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 8.2 and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q
    
def treasure_3(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 11.5 and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q
    
def treasure_4(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 14.0 and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q
    
def treasure_5(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 15.1 and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q
    
def treasure_6(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 16.1 and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q
    
def treasure_7(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 19.6 and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q
    
def treasure_8(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 20.3 and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q
    
def treasure_9(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 22.4 and p > 0:
        return 2
    elif p <= 0.0000001:
        return 4
    else:
        return q
    
def treasure_10(q, data, agent):
    env = data["env"]
    x, y = data["x"], data["y"]
    p = data["p"]
    treasure_value = env.get_map_value((x, y))
    #if treasure_value == 0:
    #    return q
    if treasure_value == 23.7 and p > 0:
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

def num_tasks4_t1():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_1)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks4_t2():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_3)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks4_t3():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_6)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks4_t4():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_9)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks5_t1():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_1)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks5_t2():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_3)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks5_t3():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_5)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks5_t4():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_7)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def num_tasks5_t5():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_9)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

def task_t10():
    dfa = DFA(0, [3], [4], [2])
    dfa.add_state(0, activate)
    dfa.add_state(1, treasure_10)
    dfa.add_state(2, just_finished)
    dfa.add_state(3, accepting)
    dfa.add_state(4, failure)
    return dfa

tasks1 = [num_tasks2_t1(), num_tasks2_t2()]
tasks2 = [num_tasks3_t1(), num_tasks3_t2(), num_tasks3_t3()]
tasks3 = [num_tasks4_t1(), num_tasks4_t2(), num_tasks4_t3(), 
          num_tasks4_t4()]
tasks4 = [num_tasks5_t1(), num_tasks5_t2(), num_tasks5_t3(), 
          num_tasks5_t4(), num_tasks5_t5()]

tasks5 = [num_tasks5_t1(), num_tasks5_t2(), num_tasks5_t3(), 
          num_tasks5_t4(), num_tasks5_t5(), task_t10()]

task_set = [tasks1, tasks2]
    
def test_model_based_pql(tasks, datfile, planning=False, num_samples=0):
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
    elif args.learning:
        dec_stapu.make_models(num_trajectories, batch_size=None, load=args.load_data, save=False)
        # Test the model outputs to see how well the models learned the environments
    if args.save_model:
        for a in range(env.num_agents):
            dec_stapu.models[a].save_model()
        
    if args.learning:
        dec_stapu.test_models(10, print_states=False)
    
    #ref_point = np.array([0] * env.num_agents + [-25] * len(tasks))
    ref_point = np.array([-100] + [0] * len(tasks))

    agent = DynaPQL(
        mamodel,
        ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay=0.999,
        final_epsilon=0.1,
        seed=1,
        log=True,
        planning=planning,
        model=dec_stapu,
        tasks=tasks,
        mcts=planning,
        collect_data=True
    )
    # first learn the STAPU model
    #stapu_state_space = np.prod(mamodel.state_shape)
    # TODO insert the other Model based training setup here as well
    # so that we can capture both data points. 
    #Training
    name = "Model-Free"
    num_episodes = 5000
    for sample in range(1):
        agent.initial_epsilon = 1.0
        agent.epsilon_decay = 0.999
        agent.final_epsilon = 0.2
        agent.reset_agent()
        pf_modelfree = agent.train(num_episodes=num_episodes, log_every=1, 
            action_eval="hypervolume", max_steps_per_episode=200, 
            data_ref=name, sample=1)
        #agent.append_to_file(datfile)

    #name = "Model-Based"
    #num_episodes = 2000
    #agent.planning = True
    #agent.mcts = True
    #agent.print_outputs = False
    #for sample in range(1):
    #    agent.initial_epsilon = 0.2
    #    agent.epsilon_decay = 1.0
    #    agent.final_epsilon = 0.1
    #    agent.reset_agent()
    #    pf_modelbased = agent.train(num_episodes=num_episodes, log_every=1, 
    #        action_eval="hypervolume", max_steps_per_episode=200, 
    #        data_ref=name, sample=0)
    #    #agent.append_to_file(datfile)


    # Save the episode data to disk
    #ep_df = pd.DataFrame(data=agent.data, columns=["episode", "name", "sample", "hv"])
    #ep_df.to_csv(f'/home/thomas/ai_projects/LearnTAP-v2/data/episode_{datfile}')

    # convert the pareto front to a numpy array and save to csv
    #data = list(pf_modelbased)
    #df = pd.DataFrame(data)
    #df.to_csv(f'/home/thomas/ai_projects/LearnTAP-v2/data/mb_pf_{datfile}', header=False)

    #data = list(pf_modelfree)
    #df = pd.DataFrame(data)
    #df.to_csv(f'/home/thomas/ai_projects/LearnTAP-v2/data/mf_pf_{datfile}', header=False)



#stem = "mb" if planning else "mf"
datfile = f"A2_T{len(tasks5)}.csv" 
sample_size=1
test_model_based_pql(tasks5, datfile, num_samples=sample_size, planning=False)
