from model_based.tutorial.dynaq import *
from utils.plotting import plot_performance

# set for reproducibility, comment out / change seed value for different results
np.random.seed(1)

# parameters needed by our policy and learning rule
params = {
  'epsilon': 0.05,  # epsilon-greedy policy
  'alpha': 0.5,  # learning rate
  'gamma': 0.8,  # temporal discount factor
  'k': 10,  # number of Dyna-Q planning steps
}

# episodes/trials
n_episodes = 500
max_steps = 1000

# environment initialization
env = QuentinsWorld()

## solve Quentin's World using Dyna-Q -- uncomment to check your solution
results = learn_environment(env, dyna_q_model_update, dyna_q_planning,
                             params, max_steps, n_episodes)
value, reward_sums, episode_steps = results

## Plot the results
plot_performance(env, value, reward_sums, n_episodes)