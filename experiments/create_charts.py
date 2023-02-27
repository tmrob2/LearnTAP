import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import seaborn as sns

def polytope_data(num_agents):
    df = pd.read_csv('data/mb-2A2T.csv', index_col=0, header=None)
    arr = df.values
    total_cost = arr[:, :num_agents].sum(axis=1)
    total_cost = np.expand_dims(total_cost, 1)
    b = arr[:, num_agents:]
    c = np.concatenate((total_cost, b), axis=1)
    return c

def create_pareto_curve():
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=20)    # legend fontsize
    plt.rc('font', size=20)          # controls default text 

    # save the data
    pf_data = pd.read_csv('/home/thomas/ai_projects/LearnTAP-v2/data/single-agent-pf.csv', index_col=0)

    fig = plt.figure()
    ax1 = fig.add_subplot()
    mbx, mby = zip(*list(pf_data["Model-Based"]))
    mfx, mfy = zip(*list(pf_data["Model-Free"]))
    ax1.scatter(mbx, mby, c="r", marker="s", facecolor='none', s=20., label="Model Based")
    ax1.scatter(mfx, mfy, c="b", marker="s", label="Model Free", s=10.)
    ax1.set_xlabel("Treasure Value")
    ax1.set_ylabel("Time Penalty")
    #ax1.set_title("Comparison of learned Pareto curves")
    plt.legend(loc="lower left")
    plt.show()

def create_polytope(num_agents):

    pts = polytope_data(num_agents)
    hull = ConvexHull(pts)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot defining corner points
    ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    plt.show()

def create_mb_mf_compare_chart(fname):
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=20)    # legend fontsize
    plt.rc('font', size=18)          # controls default text 
    df = pd.read_csv(f'/home/thomas/ai_projects/LearnTAP-v2/data/{fname}.csv', index_col=0)
    plt.figure(figsize=(10, 10))
    ax = sns.lineplot(data=df, x="episode", y="hv", hue="name", linewidth=2.5)
    #ax = df.plot(x="episode", y="hv", hue="name", lw=2.5)
    ax.set(xlabel="Epsiodes", ylabel="Hypervolume", 
        #title="Comparison of learning speed in DST environment",
    )
    plt.yscale('log')
    plt.xscale('log')
    ax.legend(title="Alg. Type", title_fontsize=20, loc="lower right")
    plt.show()


def create_mb_mf_compare_pf_card_chart(fname):
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=20)    # legend fontsize
    plt.rc('font', size=18)          # controls default text 
    df = pd.read_csv(f'/home/thomas/ai_projects/LearnTAP-v2/data/{fname}.csv', index_col=0)
    plt.figure(figsize=(10, 10))
    ax = sns.lineplot(data=df, x="episode", y="pc", hue="name", linewidth=2.5)
    ax.set(xlabel="Epsiodes", ylabel="Pareto Cardinality")
    #plt.yscale('log')
    plt.xscale('log')
    ax.legend(title="Alg. Type", title_fontsize=20, loc="lower right")
    plt.show()

def create_learning_speed(fname):
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.rc('legend', fontsize=20)    # legend fontsize
    plt.rc('font', size=13)      
    cent_df = pd.read_csv(f'/home/thomas/ai_projects/LearnTAP-v2/data/cent-loss-{fname}.csv')
    dec_df = pd.read_csv(f'/home/thomas/ai_projects/LearnTAP-v2/data/dec-loss-{fname}.csv')
    full = pd.concat([cent_df, dec_df], axis=0)
    plt.figure(figsize=(5, 5))
    ax = sns.lineplot(data=full, x="episode", y="loss", hue="model-type", linewidth=2.5)
    ax.set(xlabel="Epochs", ylabel="Loss")
    plt.yscale('log')
    #plt.xscale('log')
    ax.legend(title="Alg. Type", title_fontsize=20)
    plt.show()

#create_polytope(2)
#create_mb_mf_compare_chart("single-agent-mf-vs-mb")
#create_mb_mf_compare_pf_card_chart("episode_A2_T3")
create_learning_speed("T3")
