from typing import Optional
import torch
from typing import Callable, Optional
import numpy as np
import gymnasium as gym
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from torch.optim import Adam
from  model_based.mo.networks.nn import QObj
import torch.nn as nn
import math
import random
from collections import deque, namedtuple
from itertools import groupby
import itertools
import operator

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 10000 # Number of transitions sampled from the replay buffer
#GAMMA = 0.99
EPS_STATE = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1E-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_networks(observation_shape, n_actions, n_objs, hidden: Optional[int] = 64, lr=0.001):
    Qnet = [QObj(observation_shape, hidden, n_actions).to(device) for _ in range(n_objs)]
    QTarget = [QObj(observation_shape, hidden, n_actions).to(device) for _ in range(n_objs)]
    [QTarget[k].load_state_dict(Qnet[k].state_dict()) for k in range(n_objs)]
    Rnet = [QObj(observation_shape, hidden, n_actions).to(device) for _ in range(n_objs)]
    #RTarget = QObj(observation_shape, hidden, n_actions).to(device)
    #RTarget.load_state_dict(RTarget.state_dict())

    optimQ = [Adam(Qnet[k].parameters(), lr=lr) for k in range(n_objs)]
    optimR = [Adam(Rnet[k].parameters(), lr=lr) for k in range(n_objs)]
    return Qnet, QTarget, Rnet, optimQ, optimR


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MCTSDeepPQL(MOAgent):
    """Pareto Deep Q-learning.
    Based on Paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.
    """

    def __init__(
        self,
        env: gym.Env,
        ref_point: np.ndarray,
        num_objs: int,
        state_shape: int, 
        gamma: float = 0.8,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        final_epsilon: float = 0.1,
        seed: int = None,
        project_name: str = "MORL-baselines",
        experiment_name: str = "Pareto Q-Learning",
        tMaxRollout = 200,
        log: bool = True,
        hidden_dim: int = 64
    ):
        """Initialize the Deep Pareto Q-learning algorithm.
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
        # Networks
        Qnet, Qtarget, Rnet, optimQ, optimR = make_networks(1, env.action_space.n, hidden_dim)
        self.Qnet, self.Qtarget, self.Rnet = Qnet, Qtarget, Rnet
        self.optimQ, self.optimR = optimQ, optimR
        self.memory = ReplayMemory(10000)

        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Algorithm setup
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.ref_point = ref_point

        # MCTS Params
        self.tMax = tMaxRollout
        self.num_objectives = num_objs
        self.c = 1

        self.env: gym.Env = env
        self.num_actions = self.env.action_space.n
        low_bound = self.env.observation_space.low
        high_bound = self.env.observation_space.high
        self.env_shape = (high_bound[0] - low_bound[0] + 1, high_bound[1] - low_bound[1] + 1)
        self.num_states = np.prod(self.env_shape)
        self.num_objectives = self.env.reward_space.shape[0]
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

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

    def mcts_score_hypervolume(self, state, q_values, r_values):
        q_sets = [self.mcts_q_set(state, action, q_values, r_values) for action in range(self.num_actions)]
        #s = {tuple(vec) for vec in q_sets}
        #q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return action_scores

    def score_hypervolume(self, state):
        """Compute the action scores based upon the hypervolume metric.
        Args:
            state (int): The current state.
        Returns:
            ndarray: A score per action.
        """
        #q_arr = rmcts + qmcts
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        #s = {tuple(vec) for vec in q_sets}
        #q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return action_scores


    def select_action(self, state, score_func: Callable, d, term):
        """Select an action in the current state.
        Args:
            state (int): The current state.
            score_func (callable): A function that returns a score per action.
        Returns:
            int: The selected action.
        """
        qvalues = {}
        node = None
        UctSearch(state, self.num_actions, self.env, 2, 100, node, 100, 100)
        
        if self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.integers(self.num_actions)
        else:
            #
            sidx = int(state.detach().cpu().numpy()[0][0])
            action_scores = score_func(sidx)
            return self.rng.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())
    
    def get_q_set(self, state: int, action):
        """Compute the Q-set for a given state-action pair.
        Args:
            state (int): The current state.
            action (int): The action.
        Returns:
            A set of Q vectors.
        """
        #with torch.no_grad():
        #    nd_array = self.Qnet(state)[:, action]
        #    q_array = self.Rnet(state)[:, action] + self.gamma * nd_array
        nd_array = np.array(list(self.Q[(state, action)]))
        q_array = self.R[(state, action)] + self.gamma * nd_array
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

    def optimise(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        #non_final_reward_mask = torch.cat([s for s in batch.state if s is not None])
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        Qvalues = [self.Qnet[k](state_batch).gather(1, action_batch) 
            for k in range(self.num_objectives)]
        Rvalues = [self.Rnet[k](state_batch).gather(1, action_batch)
            for k in range(self.num_objectives)]

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        #Rvalues = torch.zeros(BATCH_SIZE, self.num_objectives, device=device)
        
        # update each of the objective networks separately
        for k in range(self.num_objectives):
            with torch.no_grad():
                next_state_values[non_final_mask] = self.Qtarget[k](non_final_next_states).max(1)[0]
                #Rvalues[non_final_mask] = self.Rtarget(state_batch).gather(2, action_batch_).squeeze()

            # Compute the expected Q values
            expected_Qvalues = (next_state_values * self.gamma) + reward_batch
            
            criterion = nn.SmoothL1Loss()
            qloss = criterion(Qvalues, expected_Qvalues)
            rloss = criterion(reward_batch, Rvalues)

            self.optimQ[k].zero_grad()
            qloss.backward()
            torch.nn.utils.clip_grad_value_(self.Qnet[k].parameters(), 100)
            self.optimQ.step()

            self.optimR.zero_grad()
            rloss.backward()
            torch.nn.utils.clip_grad_value_(self.Rnet[k].parameters(), 100)
            self.optimR.step()

        #print(f"qloss: {qloss.detach().cpu().numpy()}, rloss: {rloss.detach().cpu().numpy()}")
        
    def train(
        self, 
        num_episodes: Optional[int] = 3000, 
        log_every: Optional[int] = 100, 
        action_eval: Optional[str] = "hypervolume",
    ):
        """Learn the Pareto front.
        Args:
            num_episodes (int, optional): The number of episodes to train for.
            log_every (int, optional): Log the results every number of episodes. (Default value = 100)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')
        Returns:
            Set: The final Pareto front.
        """

        #model = np.nan * np.zeros((self.num_states, self.num_actions, self.reward_dim + 1))

        if action_eval == "hypervolume":
            score_func = self.score_hypervolume
        elif action_eval == "pareto_cardinality":
            score_func = self.score_pareto_cardinality
        else:
            raise Exception("No other method implemented yet")

        for episode in range(num_episodes):
            #self._reset_tree()
            if episode % log_every == 0:
                print(f"Training episode {episode + 1}")

            state, _ = self.env.reset()
            state = int(np.ravel_multi_index(state, self.env_shape))
            done = False
            terminated = False

            # This bit has to be replaced by deep Q-learnin algo
            while not done:
                #state_ = torch.from_numpy(np.tile(np.array([state], dtype=np.float32), (self.num_objectives, 1))).to(device)
                action = self.select_action(state, score_func, 0, terminated)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = int(np.ravel_multi_index(next_state, self.env_shape))
                    #next_state_ = torch.from_numpy(np.tile(np.array([next_state], dtype=np.float32), (self.num_objectives, 1))).to(device)
                    reward_ = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
                    #next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward_)
                
                state = next_state

                # perform one step optimsiation
                self.optimise()

                target_qnet_state_dict = self.Qtarget.state_dict()
                #target_rnet_state_dict = self.Rtarget.state_dict()
                policy_qnet_state_dict = self.Qnet.state_dict()
                #policy_rnet_state_dict = self.Rnet.state_dict()

                for key in policy_qnet_state_dict:
                    target_qnet_state_dict[key] = policy_qnet_state_dict[key] * TAU + target_qnet_state_dict[key] * (1-TAU)
                #for key in policy_rnet_state_dict:
                #    target_rnet_state_dict[key] = policy_rnet_state_dict[key] * TAU + target_rnet_state_dict[key] * (1-TAU)
                self.Qtarget.load_state_dict(target_qnet_state_dict)
                #self.Rtarget.load_state_dict(target_rnet_state_dict)

            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

            if self.log and episode % log_every == 0:
                pf = self.get_local_pcs(state=0)
                value = hypervolume(self.ref_point, list(pf))
                print(f"Hypervolume in episode {episode}: {value}")
                #self.writer.add_scalar("train/hypervolume", value, episode)

        return self.get_local_pcs(state=0)

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


class MCTSNode:
    id_iter = itertools.count()

    def __init__(self, params, done, depth, num_objectives):
        self.params = params
        self.children = {}
        self.parent = None
        self.Q = {tuple([0.] * num_objectives)}
        self.N = 0
        self.id = next(MCTSNode.id_iter)
        self.done = done
        self.depth = depth
        self.reward = np.array([0.] * num_objectives)
        self.num_objectives = num_objectives

    def add_child(self, action, child):
        self.children[action] = child


def UctSearch(params, n_actions, environment, n_objs, iterations=100, 
        node=None,  C_p=None, lookahead_target=None):
    if C_p == None:
        C_p = 200
    if lookahead_target == None:
        lookahead_target = 200
    if node == None:
        root_node = MCTSNode(params, False, 0, n_objs)
    else:
        root_node = node

    counter = 0
    max_depth = 0
    ix = 0
    while True:
        v = TreePolicy(root_node, C_p, n_actions, environment, n_objs)
        max_depth = max(v.depth - root_node.depth, max_depth)
        # The reward returned will be multi-objecjtive
        Delta = DefaultPolicy(v, environment)
        Backup(v, Delta, root_node)
        counter += 1
        ix += 1
        if ix > iterations:
            break
    if max_depth < lookahead_target:
        C_p = C_p - 1
    else:
        C_p = C_p + 1
    print(
        f"### max_depth: {max_depth:03}, lookahead_target: {lookahead_target:03} ")
    print(f"### C_p: {C_p} ")
    print("### Maximal depth considered: ", max_depth)
    for action, child in sorted(root_node.children.items()):
        print(
            f"### action: {action}, Q: {int(child.Q):08}, N: {child.N:08}, Q/N: {child.Q/child.N:07.2f}")

    best_child = max(root_node.children.values(), key=lambda x: x.N)
    best_child_action = best_child.action
    print(f"### predicted state: {best_child.params}")
    print(f"### chosen action: {best_child_action}")

    best_child_node = max(root_node.children.values(), key=lambda x: x.N)
    #return (best_child_node.action, best_child_node, C_p)


def TreePolicy(node, C_p, n_actions, environment, n_objs):
    while not node.done:
        if len(node.children) < n_actions:
            return Expand(node, n_actions, environment, n_objs)
        else:
            node = BestChild(node, C_p)
    return node


def Expand(node, n_actions, environment, n_objs):
    exp_env = environment
    exp_env.reset()
    exp_env.unwrapped.state = node.params

    unchosen_actions = list(
        filter(lambda action: not action in node.children.keys(), range(n_actions)))
    a = random.choice(unchosen_actions)
    params, _, done, _, _ = exp_env.step(a)
    child_node = MCTSNode(params, done, node.depth+1, n_objs)
    child_node.parent = node
    node.children[a] = child_node
    child_node.action = a
    return child_node


def BestChild(node, c, random=False):
    if random:
        child_values = {child: child.Q/child.N + c *
                        math.sqrt(2*math.log(node.N)/child.N) for child in node.children}
        mv = max(child_values.values())
        am = random.choice([k for (k, v) in child_values.items() if v == mv])
    else:
        am = max(node.children.values(), key=lambda v_prime: v_prime.Q /
                 v_prime.N + c*math.sqrt(2*math.log(node.N)/v_prime.N))
    return am


def DefaultPolicy(node, environment):
    exp_env = environment
    exp_env.reset()
    exp_env.unwrapped.state = node.params

    done = node.done
    reward = node.reward

    while not done:
        random_action = random.choice([0, 1])
        _, step_reward, done, _, _ = exp_env.step(random_action)
        reward += (step_reward - reward) / node.N

    return reward


def Backup(node, Delta, root_node):
    while not node is root_node.parent:
        node.N += 1
        node.Q = np.array(list(node.Q)) + Delta
        node = node.parent