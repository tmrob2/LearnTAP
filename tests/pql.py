from model_based.ma_envs.dst import DEFAULT_MAP
from model_based.ma_envs.dst import DeepSeaTreasure
import numpy as np
from model_based.mo.agents.mo_dyna_pql import DynaPQL
import matplotlib.pyplot as plt

def test_model_based_pql():
    #env_id = "deep-sea-treasure-v0"
    #env = mo_gym.make(env_id, dst_map=DEFAULT_MAP)
    env = DeepSeaTreasure(None, DEFAULT_MAP)
    ref_point = np.array([0, -25])
    num_episodes = 700
    agent = DynaPQL(
        env,
        ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay=0.997,
        final_epsilon=0.2,
        seed=1,
        log=True
    )

    # Training
    pf = agent.train(num_episodes=num_episodes, log_every=100, action_eval="hypervolume", k=20)
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
