from typing import Callable, Optional, List
import numpy as np
from model_based.tap_models.stapu_env import Progress
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from model_based.mo.planning.mamo.MCTS import MAMOMCTSNode
from utils.dfa import DFA
import torch
from pymoo.indicators.hv import HV
import random
import tqdm
import pandas as pd


class DynaPQL(MOAgent):
    """Pareto Q-learning.
    Tabular method relying on pareto pruning.
    Paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.
    """

    def __init__(
        self,
        env,
        ref_point: np.ndarray,
        gamma: float = 1.0, #0.8,
        initial_epsilon: float = 0.2,
        epsilon_decay: float = 0.99,
        final_epsilon: float = 0.1,
        seed: int = None,
        project_name: str = "MORL-baselines",
        experiment_name: str = "Pareto Q-Learning",
        log: bool = True,
        tasks: List[DFA] = [],
        planning = False,
        model = None,
        mcts = False,
        print_outputs = False,
        collect_data = False
    ):
        """Initialize the Pareto Q-learning algorithm.
        Args:
            env: The environment.
            ref_point: The reference point for the hypervolume metric.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            epsilon_decay: The epsilon decay rate.
            final_epsilon: The final epsilon value.
            seed: The random seed.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            log: Whether to log or not.
        """
        super().__init__(env)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.model = model

        # Algorithm setup
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.ref_point = ref_point

        self.num_actions = self.env.action_space.n
        #low_bound = self.env.observation_space.low
        #high_bound = self.env.observation_space.high
        #self.env_shape = (high_bound[0] - low_bound[0] + 1, high_bound[1] - low_bound[1] + 1)
        self.env = env
        self.env_shape = env.state_shape
        self.num_states = np.prod(self.env_shape)
        self.num_objectives = self.env.reward_space.shape[0]
        self.num_agents = self.env.env.num_agents
        self.counts = np.zeros((self.num_states, self.num_actions, self.num_agents + len(tasks)))
        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.print_outputs = print_outputs
        self.collect_data = collect_data
        self.data = []

        #if self.log:
        #    self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name)
        # TODO remove this and replace with model simulation of done or not
        # i.e. use a model to learn the DFA so we don't store them in the 
        # algorithm
        self.tasks = tasks # The DFAs to be included in checking

        # model data for supervised learning
        self.model_data = None
        self.target_data = None

        # planning
        self.planning = planning
        if planning:
            self.model = model
        self.mcts = mcts

    def collect_episode_data(self, episode, hv, name, pc, sample=0):
        self.data.append([episode, name, sample, hv, pc])

    def create_data_file(self, fname):
        df = pd.DataFrame(data=[], columns=["episode", "name", "sample", "hv"])
        df.to_csv(f'/home/thomas/ai_projects/LearnTAP-v2/data/episode_{fname}', index=False)
        print("Created csv file..")

    def append_to_file(self, fname):
        ep_df = pd.DataFrame(data=self.data, columns=["episode", "name", "sample", "hv", "pc"])
        ep_df.to_csv(f'/home/thomas/ai_projects/LearnTAP-v2/data/episode_{fname}', index=False, mode='a', header=False)
        print("Saved data to csv..")

    def reset_agent(self):
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

    def get_config(self) -> dict:
        """Get the configuration dictionary.
        Returns:
            Dict: A dictionary of parameters and values.
        """
        return {
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
            "seed": self.seed,
        }

    def agent_adjusted_hypervolume(self, qset):
        # Q set is a list of tuples
        # we want the team score of the agents and the task scores
        q_vecs = np.array(qset)
        # remove all non zero set entries
        q_vecs = q_vecs[~np.all(q_vecs==0, axis=1)]
        b = self.team_score(q_vecs[:, :self.num_agents])
        b_ = np.expand_dims(b, 1)
        adjusted = np.concatenate((b_, q_vecs[:, self.num_agents:]), axis=1)
        # adjust zero values to small value
        adjusted[adjusted == 0] = 0.01
        return HV(ref_point=self.ref_point * -1)(adjusted * -1)


    @staticmethod
    def team_score(agent_scores):
        return np.sum(agent_scores, axis=1)

    def score_hypervolume(self, state: int):
        """Compute the action scores based upon the hypervolume metric.
        Args:
            state (int): The current state.
        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        #action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        action_scores = [self.agent_adjusted_hypervolume(list(q_set)) for q_set in q_sets]
        return action_scores

    def score_pareto_cardinality(self, state: int):
        """Compute the action scores based upon the Pareto cardinality metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        non_dominated = get_non_dominated(candidates)
        scores = np.zeros(self.num_actions)

        for vec in non_dominated:
            for action, q_set in enumerate(q_sets):
                if vec in q_set:
                    scores[action] += 1

        return scores

    def select_action(self, state: int, score_func: Callable):
        """Select an action in the current state.
        Args:
            state (int): The current state.
            score_func (callable): A function that returns a score per action.
        Returns:
            int: The selected action.
        """
        if self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.rng.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())
    
    def get_q_set(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.
        Args:
            state (int): The current state.
            action (int): The action.
        Returns:
            A set of Q vectors.
        """
        nd_array = np.array(list(self.non_dominated[state][action]))
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}

    def calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state.
        Args:
            state (int): The current state.
        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        candidates = set().union(*[self.get_q_set(state, action) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated

    def enabled_actions(self, state):
        valid_actions = []
        actions = list(range(4))
        for action in actions:
            next_state = state[1:3] + self.env.dir[action]
            if self.env.is_valid_state(next_state):
                valid_actions.append(action)        
        Q = state[3:]
        if all([t.model_is_idle(Q[j]) for j, t in enumerate(self.tasks)]):
            valid_actions.append(4)
        for j, t in enumerate(self.tasks):
            if t.model_not_started(Q[j]):
                valid_actions.append(6 + j)
        return valid_actions

    def train(
        self, 
        num_episodes: Optional[int] = 3000, 
        log_every: Optional[int] = 100, 
        action_eval: Optional[str] = "hypervolume",
        max_steps_per_episode=100,
        data_ref: Optional[str] = "data",
        sample=0
    ):
        """Learn the Pareto front.
        Args:
            num_episodes (int, optional): The number of episodes to train for.
            log_every (int, optional): Log the results every number of episodes. (Default value = 100)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')
        Returns:
            Set: The final Pareto front.
        """

        if action_eval == "hypervolume":
            score_func = self.score_hypervolume
        elif action_eval == "pareto_cardinality":
            score_func = self.score_pareto_cardinality
        else:
            raise Exception("No other method implemented yet")
        ep_count = 0
        max_steps = max_steps_per_episode
        with tqdm.trange(num_episodes) as t:
            for episode in t:
                steps = 0
                #if episode % log_every == 0:
                #    print(f"Training episode {episode + 1}")

                state_, _ = self.env.reset()
                state = int(np.ravel_multi_index(state_, self.env_shape))
                terminated = False
                truncated = False

                while not (terminated or truncated) and steps < max_steps:
                    Q = state_[3:]
                    if self.planning and self.mcts:
                        planning_state = state_.astype(np.float32)
                        planning_state[1:3] = planning_state[1:3] / 10.
                        root = MAMOMCTSNode(
                            planning_state, self.num_actions, self.model, self.tasks, 
                            self.env.state_shape, self.ref_point, self.seed,
                            score_func, self.calc_non_dominated,
                            self.non_dominated, self.avg_reward, self.counts, p=self.env.p)
                        if self.rng.uniform(0, 1) < self.epsilon:
                            actions = self.enabled_actions(state_)
                            #action = self.rng.integers(self.num_actions)
                            action = random.choice(actions)
                        else:
                            action = root.best_action()
                        #action = root.best_action()
                    else:
                        action = self.select_action(state, score_func)
                    #action_scores = score_func(state)
                    #action = self.rng.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())
                    #task_count_mask = [
                    #    0 if t.is_finished() or not t.in_progress() else 1 for t in self.tasks
                    #]
                    #agent_count_mask = [
                    #    1 if self.env.env.agents[i].active else 0 for i in range(self.env.env.num_agents)
                    #]
                    # To a TD step with the real environment
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    #print("s", state_, "a", action, "s'", next_state)
                    state_ = next_state
                    next_state = int(np.ravel_multi_index(next_state, self.env_shape))
                    self.counts[state, action] += 1 #np.array(agent_count_mask + task_count_mask)
                    self.non_dominated[state][action] = self.calc_non_dominated(next_state)
                    #print(self.get_local_pcs(state=0))
                    a = reward - self.avg_reward[state, action]
                    b = self.counts[state, action]
                    self.avg_reward[state, action] +=  a / b #np.divide(a, b, out=np.zeros_like(a), where=b!=0)
                    #if any(q == 2 for q in Q):
                    value = self.env.env.get_map_value((int(state_[1]), int(state_[2])))
                    #if value > 0:
                    #print(f"state", state, f"R[{state}, {action}] {np.around(self.avg_reward[state, action], 3)}", f"C {self.counts[state, action]}, Q: {Q}, (x, y): {(int(state_[1]), int(state_[2]))} value: {value}, {self.env.p}")
                    
                    state = next_state
                    steps += 1

                self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

                if self.log and episode % log_every == 0:
                    pf = self.get_local_pcs(state=0)
                    #print(pf)
                    
                    value = self.agent_adjusted_hypervolume(list(pf))
                    name = "MF PQL" if not self.planning else "MB PQL"
                    t.set_description(name)
                    printvalue = f"{value:.3f}"
                    t.set_postfix(e=episode, HV=printvalue, PC=len(pf))
                    if self.print_outputs: 
                        print(pf)
                        print(f"Hypervolume in episode {episode}: {value}")

                    if self.collect_data:
                        self.collect_episode_data(episode, value, data_ref, len(pf), sample)
                ep_count += 1
        return self.get_local_pcs(state=0)

    def enabled_actions(self, state):
        actions = list(range(4))
        Q = state[3:]
        for j in range(len(self.tasks)):
            if self.tasks[j].model_not_started(Q[j]):
                actions.append(6 + j)
        if all(t.model_is_idle(Q[j]) for j, t in enumerate(self.tasks)) and \
            not all(t.model_is_finished(Q[j]) for j, t in enumerate(self.tasks)):
            actions.append(4)
        return actions

    def make_table_index(self, state):
        env_coors = (state[1:3] * (np.array(self.env_shape)-1.)[1:3]).astype(np.int32)
        state_int = [int(state[0]), env_coors[0], env_coors[1]]
        state_int.extend(map(lambda x: int(x), list(state[3:])))
        state_int = tuple(state_int)
        sidx = np.ravel_multi_index(state_int, self.env_shape)
        return sidx
            
    def track_policy(self, vec):
        """Track a policy from its return vector.
        Args:
            vec (array_like): The return vector to track.
        """
        target = np.array(vec)
        state, _ = self.env.reset()
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)

        while not (terminated or truncated):
            state = np.ravel_multi_index(state, self.env_shape)
            new_target = False

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state, action]
                non_dominated_set = self.non_dominated[state][action]
                for q in non_dominated_set:
                    q = np.array(q)
                    if np.all(self.gamma * q + im_rew == target):
                        state, reward, terminated, truncated, _ = self.env.step(action)
                        total_rew += reward
                        target = q
                        new_target = True
                        break

                if new_target:
                    break

        return total_rew

    def get_local_pcs(self, state: int = 0):
        """Collect the local PCS in a given state.
        Args:
            state (int): The state to get a local PCS for. (Default value = 0)
        Returns:
            Set: A set of Pareto optimal vectors.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        return get_non_dominated(candidates)