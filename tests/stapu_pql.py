from mo_gym.deep_sea_treasure.deep_sea_treasure import DEFAULT_MAP
import numpy as np
from model_based.mo.agents.mamo_dyna_pql import DynaPQL
from model_based.ma_envs.multi_agent_deep_sea_treasure.ma_deep_sea_treasure import DeepSeaTreasure
from model_based.tap_models.stapu_env import STAPU
import matplotlib.pyplot as plt
from utils.dfa import DFA

# define a simple test task to make sure that this is working
# A DFA is defined with tranisition functions 
# Informally, let the task be find two pieces of treasure
Qstates = [0, 1, 2] # 0 is initial, 1 is found treasure, 2 is fail

def treasure_found(q, env, agent):
    treasure_value = env.get_map_value(env.agents[agent].current_state)
    if treasure_value == 0:
        return q
    elif treasure_value > 20:
        return 1
    else:
        return 2

# Define the sink states
def accepting(q, env, agent):
    return 1

def failure(q, env, agent):
    return 2

def find_a_treasure():
    dfa = DFA(0, [1], [2])
    dfa.add_state(0, treasure_found)
    dfa.add_state(1, accepting)
    dfa.add_state(2, failure)
    return dfa

tasks = [find_a_treasure()]

def test_model_based_pql():
    env = DeepSeaTreasure(render_mode=None)
    mamodel = STAPU(env, tasks)
    #ref_point = np.array([0] * env.num_agents + [-25] * len(tasks))
    ref_point = np.array([0, 0, -25])
    num_episodes = 1000
    agent = DynaPQL(
        mamodel,
        ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay=0.997,
        final_epsilon=0.2,
        seed=1,
        log=True 
    )

    # Training
    pf = agent.train(
        num_episodes=num_episodes, log_every=100, action_eval="hypervolume", k=50
    )
    assert len(pf) > 0
    print(pf)

    x, y, z = zip(*list(pf))
    # put this one on the first sub-figure
    plt.scatter(x, z)
    # put this one on the second sub-figure
    plt.scatter(y, z)
    plt.show()

    # Policy following
    target = np.array(pf.pop())
    print(f"Tracking {target}")
    # Fix tracking with STAPU environment
    #reward = agent.track_policy(target)
    print(f"Obtained {reward}")

test_model_based_pql()
