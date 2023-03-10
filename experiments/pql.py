from mo_gym.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure, DEFAULT_MAP
import numpy as np
from model_based.mo.agents.model_pql import PQL
from model_based.mo.planning.single_agent_mo.single_agent_mcts import SAMOModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_data(self, data):
    sns.set_style('darkgrid')
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('legend', fontsize=13)
    plt.rc('font', size=13)
    plt.figure(figsize=(8, 4))
    ax = sns.lineplot(data=data, linewidth=2.5)
    ax.set(xlabel="Episodes", ylabel="Hypervolume", 
        title="Comparison of Model Based and Model Free PQL Biobjective DST")
    ax.legend(title="Alg. Type")
    plt.show()

def test_model_based_pql():
    #env_id = "deep-sea-treasure-v0"
    #env = mo_gym.make(env_id, dst_map=DEFAULT_MAP)
    env_ = DeepSeaTreasure(None, DEFAULT_MAP)
    model = SAMOModel(env_)
    env = DeepSeaTreasure(None, DEFAULT_MAP)
    ref_point = np.array([0, -25])
    agent = PQL(
        env,
        ref_point,
        gamma=1.0,
        initial_epsilon=1.,
        epsilon_decay=0.999,
        final_epsilon=0.2,
        seed=1,
        log=True,
        planning=False,
        model=model,
        print_outputs=False,
        collect_data=True
    )
    # Construct a new dataframe to hold the model based and model free data
    # Training
    # Model based
    num_samples = 30
    name = f"Model-Based"
    #agent.make_new_dataset(name)
    for s in range(1):
        num_episodes = 10
        agent.planning = True
        agent.initial_epsilon = 0.5
        agent.epsilon_decay = 0.9
        agent.final_epsilon= 0.1
        agent.reset_agent()
        pf_model_based = agent.train(num_episodes=num_episodes, log_every=1, action_eval="hypervolume", data_ref=name, sample=s)
    # Model free 
    name = f"Model-Free"
    #agent.make_new_dataset(name)
    for s in range(1):
        num_episodes = 2000
        agent.planning= False
        agent.initial_epsilon = 1.
        agent.epsilon_decay = 0.999
        agent.final_epsilon = 0.2
        agent.reset_agent()
        pf_model_free = agent.train(num_episodes=num_episodes, log_every=1, action_eval="hypervolume", data_ref=name, sample=s)

    df = pd.DataFrame(data=agent.data, columns=["episode", "name", "sample", "hv"])
    
    #assert len(pf) > 0
    #print(pf)

    #x, y = zip(*list(pf))
    #plt.scatter(x, y)
    #plt.show()
    pf_data = {"Model-Free": pf_model_free, "Model-Based": pf_model_based}
    ## Policy following
    #target = np.array(pf_model_free.pop())
    #print(f"Tracking {target}")
    #reward = agent.track_policy(target)
    #print(f"Obtained {reward}")

    # get the episode hypervolume data
    #sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=20)    # legend fontsize
    plt.rc('font', size=20)          # controls default text 

    # save the data
    #df.to_csv('/home/thomas/ai_projects/LearnTAP-v2/data/single-agent-mf-vs-mb.csv')
    #plt.figure(figsize=(10, 10))
    #ax = sns.lineplot(data=df, x="episode", y="hv", hue="name", linewidth=2.5)
    #ax.set(xlabel="Epsiodes", ylabel="Hypervolume", 
    #    #title="Comparison of learning speed in DST environment",
    #)
    #plt.xscale('log')
    #ax.legend(title="Alg. Type", title_fontsize=20)
    #plt.show()

    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    mbx, mby = zip(*list(pf_data["Model-Based"]))
    mfx, mfy = zip(*list(pf_data["Model-Free"]))
    ax.scatter(mbx, mby, c="r", marker="s", facecolor='none', s=200., label="Model Based")
    ax.scatter(mfx, mfy, c="b", marker="*", label="Model Free", s=120.)
    ax.set_xlabel("Treasure Value")
    ax.set_ylabel("Time Penalty")
    #ax1.set_title("Comparison of learned Pareto curves")
    ax.legend(title="Alg. Type", title_fontsize=20, loc="lower left")
    plt.show()


test_model_based_pql()
