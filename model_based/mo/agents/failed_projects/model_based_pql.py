from typing import Callable, Optional, List
import numpy as np
from model_based.tap_models.stapu_env import Progress
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from model_based.mo.planning.MCTS import MAMOMCTSNode
from utils.dfa import DFA
from copy import deepcopy
from pymoo.indicators.hv import HV

import torch

class DynaPQL(MOAgent):
    """Pareto Q-learning.
    Tabular method relying on pareto pruning.
    Paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.
    """

    def __init__(
        self,
        env,
        model,
        ref_point: np.ndarray,
        gamma: float = 0.8,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        final_epsilon: float = 0.1,
        seed: int = None,
        project_name: str = "MORL-baselines",
        experiment_name: str = "Pareto Q-Learning",
        log: bool = True,
        tasks: List[DFA] = [],
        model_data = None,
        target_data = None
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
        self.simulation_env = deepcopy(env)
        self.env_shape = env.state_shape
        self.num_states = np.prod(self.env_shape)
        self.num_objectives = self.env.reward_space.shape[0]
        self.num_agents = self.env.env.num_agents
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        #if self.log:
        #    self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name)
        # TODO remove this and replace with model simulation of done or not
        # i.e. use a model to learn the DFA so we don't store them in the 
        # algorithm
        self.tasks = tasks # The DFAs to be included in checking

        # model data for supervised learning
        self.model_data = model_data
        self.target_data = target_data

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


    def train(
        self, 
        num_episodes: Optional[int] = 3000, 
        log_every: Optional[int] = 100, 
        action_eval: Optional[str] = "hypervolume"
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
        for episode in range(num_episodes):
            if episode % log_every == 0:
                print(f"Training episode {episode + 1}")
                #print(self.get_local_pcs())

            state, _ = self.env.reset()
            int_states = (state * (np.array(self.env_shape) - 1)).astype(np.int32)
            state_idx = int(np.ravel_multi_index(int_states, self.env_shape))
            terminated = False
            truncated = False
            p = 1.

            while not (terminated or truncated):
                #action = self.select_action(state, score_func)
                # We conduct planning here with MCTS and select the best action
                # simultaneously
                #print("MCTS root state", state)
                root = MAMOMCTSNode(
                    state, self.num_actions, self.model, self.tasks, 
                    self.env.state_shape, self.ref_point, self.seed,
                    score_func, self.calc_non_dominated,
                    self.non_dominated, self.avg_reward, self.counts, p=1.)
                if self.rng.uniform(0, 1) < self.epsilon:
                    action = self.rng.integers(self.num_actions)
                else:
                    action = root.best_action()
                # after the planning step we need to relearn the model
                # using supervised learning
                # To a TD step with the real environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                #index_next_state = (np.array(next_state) * (np.array(self.env.state_shape) - 1)).astype(np.int32)
                #print(next_state)
                next_state = int(np.ravel_multi_index(next_state.astype(np.int32), self.env_shape))
                #print(next_state)
                #self.counts[state_idx, action] += 1
                #self.non_dominated[state_idx][action] = self.calc_non_dominated(next_state)
                
                #self.avg_reward[state_idx, action] += (reward - self.avg_reward[state_idx, action]) / self.counts[state_idx, action]
                # then there is the model update part here
                #model[state, action][:self.reward_dim] = self.avg_reward[state, action]
                #model[state, action][-1] = next_state

                # and then execute the planner
                #self.planning_step(model, k)
                Q = state[3:]
                value = self.env.env.get_map_value((int(state[1]), int(state[2])))
                #print(f"state", state, f"R[{state_idx}, {action}] {np.around(self.avg_reward[state_idx, action], 3)}", f"C {self.counts[state_idx, action]}, Q: {Q}")
                print(f"state", state, 
                    f"R[{state_idx}, {action}] {np.around(self.avg_reward[state_idx, action], 3)}", 
                    f"C {self.counts[state_idx, action]}", 
                    f"Q: {Q}, (x, y): {(int(state[1] * 10), int(state[2] * 10))} value: {value}, {self.env.p}")
                state_idx = next_state
                sprime = np.array(np.unravel_index(state_idx, self.env.state_shape)).astype(np.float32)
                sprime = np.array([sprime[0]] + list(sprime[1:3] * 0.1) + list(sprime[3:]))
                #sprime = np.array(np.unravel_index(state_idx, self.env.state_shape)) / (np.array(self.env.state_shape) - 1)
                #print(f"current state: {state}, planned best action: {action}, next_state: {sprime}")
                #print("ND", self.non_dominated[state_idx][action])
                state = sprime

            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

            if self.log and episode % log_every == 0:
                pf = self.get_local_pcs(state=0)
                print(pf)
                
                value = self.agent_adjusted_hypervolume(list(pf))
                print(f"Score in episode {episode}: {value}")
                #self.writer.add_scalar("train/hypervolume", value, episode)
            ep_count += 1
        return self.get_local_pcs(state=0)

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