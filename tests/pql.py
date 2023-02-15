from mo_gym.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure, DEFAULT_MAP
import numpy as np
from model_based.mo.agents.model_pql import PQL
from model_based.mo.planning.single_agent_mo.single_agent_mcts import SAMOModel
import matplotlib.pyplot as plt

def test_model_based_pql():
    #env_id = "deep-sea-treasure-v0"
    #env = mo_gym.make(env_id, dst_map=DEFAULT_MAP)
    env_ = DeepSeaTreasure(None, DEFAULT_MAP)
    model = SAMOModel(env_)
    env = DeepSeaTreasure(None, DEFAULT_MAP)
    ref_point = np.array([0, -25])
    num_episodes = 2000
    agent = PQL(
        env,
        ref_point,
        gamma=1.0,
        initial_epsilon=0.5,
        epsilon_decay=0.99,
        final_epsilon=0.2,
        seed=1,
        log=True,
        planning=True,
        model=model
    )

    # Training
    pf = agent.train(num_episodes=num_episodes, log_every=1, action_eval="hypervolume")
    assert len(pf) > 0
    print(pf)

    x, y = zip(*list(pf))
    plt.scatter(x, y)
    plt.show()

    # Policy following
    target = np.array(pf.pop())
    print(f"Tracking {target}")
    reward = agent.track_policy(target)
    print(f"Obtained {reward}")

test_model_based_pql()
