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
import operator

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128*10 # Number of transitions sampled from the replay buffer
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
        # simulate 50 trees
        self._simulate(state, d, term)

        
        if self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.integers(self.num_actions)
        else:
            #
            sidx = int(state.detach().cpu().numpy())
            action_scores = score_func(sidx)
            return self.rng.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())

    def mcts_q_set(self, state: int, action, qvalues, rvalues):
        if not qvalues:
            nd_array = np.array(list({(0., 0.)}))
        else:
            nd_array = np.array(qvalues[action])
        q_array = rvalues[(state, action)] + self.gamma * nd_array
        try:
            return {tuple(vec) for vec in q_array}
        except:
            print("error")
    
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

    def tree_non_dominated(self, state, qvalues, rvalues):
        candidates = set().union(*[self.mcts_q_set(state, action, qvalues, rvalues) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated

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
        action_batch = torch.from_numpy(np.array(list(batch.action))).to(device)
        reward_batch = torch.cat(batch.reward)
        Qvalues = [self.Qnet[k](state_batch.unsqueeze(1)).gather(1, action_batch.unsqueeze(1)) 
            for k in range(self.num_objectives)]
        Rvalues = [self.Rnet[k](state_batch.unsqueeze(1)).gather(1, action_batch.unsqueeze(1))
            for k in range(self.num_objectives)]

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        #Rvalues = torch.zeros(BATCH_SIZE, self.num_objectives, device=device)
        
        # update each of the objective networks separately
        for k in range(self.num_objectives):
            with torch.no_grad():
                next_state_values[non_final_mask] = self.Qtarget[k](non_final_next_states.unsqueeze(1)).max(1)[0]
                #Rvalues[non_final_mask] = self.Rtarget(state_batch).gather(2, action_batch_).squeeze()

            # Compute the expected Q values
            expected_Qvalues = (next_state_values * self.gamma) + reward_batch[:, k]
            
            criterion = nn.SmoothL1Loss()
            qloss = criterion(Qvalues[k].squeeze(), expected_Qvalues)
            rloss = criterion(reward_batch[:, k], Rvalues[k].squeeze())

            self.optimQ[k].zero_grad()
            qloss.backward()
            torch.nn.utils.clip_grad_value_(self.Qnet[k].parameters(), 100)
            self.optimQ[k].step()

            self.optimR[k].zero_grad()
            rloss.backward()
            torch.nn.utils.clip_grad_value_(self.Rnet[k].parameters(), 100)
            self.optimR[k].step()
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

        self._reset_tree()

        for episode in range(num_episodes):
            #self._reset_tree()
            if episode % log_every == 0:
                #counts = [len([item[1:] for item in v]) for k,v in groupby([item for item in self.Q.items()], operator.itemgetter(0))]
                #if counts:
                #    print(f"max counts {max(counts)}")
                    #print(self.Q)
                if 0 in self.Tree:
                    print([self.Q[(0, a)] for a in range(self.num_actions)])
                    #print(self.get_local_pcs(state=0))
                    pf = self.get_local_pcs(state=0)
                    value = hypervolume(self.ref_point, list(pf))
                    print(f"Hypervolume in episode {episode}: {value}, pf points: {len(pf)}")
                    #self.writer.add_scalar("train/hypervolume", value, episode)
                    #self._reset_tree()

                print(f"Training episode {episode + 1}")

            state, _ = self.env.reset()
            state = int(np.ravel_multi_index(state, self.env_shape))
            state = torch.from_numpy(np.array([state], dtype=np.float32)).to(device)
            done = False
            terminated = False

            # This bit has to be replaced by deep Q-learnin algo
            while not done:
                action = self.select_action(state, score_func, 0, terminated)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                #print(next_state)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = int(np.ravel_multi_index(next_state, self.env_shape))
                    next_state = torch.from_numpy(np.array([next_state], dtype=np.float32)).to(device)
                    reward_ = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
                    #next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward_)
                
                state = next_state

                # perform one step optimsiation
                self.optimise()

                # update the model params
                for k in range(self.num_objectives):
                    target_qnet_state_dict = self.Qtarget[k].state_dict()
                    #target_rnet_state_dict = self.Rtarget.state_dict()
                    policy_qnet_state_dict = self.Qnet[k].state_dict()
                    #policy_rnet_state_dict = self.Rnet.state_dict()

                    for key in policy_qnet_state_dict:
                        target_qnet_state_dict[key] = policy_qnet_state_dict[key] * TAU + target_qnet_state_dict[key] * (1-TAU)
                    #for key in policy_rnet_state_dict:
                    #    target_rnet_state_dict[key] = policy_rnet_state_dict[key] * TAU + target_rnet_state_dict[key] * (1-TAU)
                    self.Qtarget[k].load_state_dict(target_qnet_state_dict)
                    #self.Rtarget.load_state_dict(target_rnet_state_dict)

            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

        return self.get_local_pcs(state=0)

    def _reset_tree(self):
        self.Tree = set()
        self.Nsa = {}
        self.Ns = {}
        self.Q = {}
        self.R = {}

    def _simulate(self, s, d, term):
        sidx = int(s.detach().cpu().numpy())
        #print(s.device)
        if term:
            return np.array([0.] * self.num_objectives)
        if d == self.tMax: # the tree depth is 0 
            return self._rollout(s, self.tMax, sidx)
        ##
        if sidx not in self.Tree:
            for a in range(self.num_actions):
                with torch.no_grad():
                    self.Nsa[(sidx, a)], self.Ns[sidx], self.Q[(sidx, a)] = 1, 1, {tuple([float(self.Qnet[k](s)[a].detach().cpu().numpy()) for k in range(self.num_objectives)])}
                    self.R[(sidx, a)] = np.array([float(self.Rnet[k](s)[a].detach().cpu().numpy()) for k in range(self.num_objectives)])
            self.Tree.add(sidx)
            return self._rollout(s, term, sidx)
        ##
        qa_tab = [np.array((list(self.Q[(sidx,a)])))+self.c*math.sqrt(math.log(self.Ns[sidx])/(1e-5 + self.Nsa[(sidx,a)])) for a in range(self.num_actions)]
        #qa_tab = [self.c*math.sqrt(math.log(self.Ns[sidx])/(1e-5 + self.Nsa[(sidx,a)])) for a in range(self.num_actions)]
        a = np.argmax(self.mcts_score_hypervolume(sidx, qa_tab, self.R))
        # choose a random action
        #a = random.choice(list(range(self.num_actions)))
        #a = np.argmax(qa_tab)
        # update the Q values
        # get the next state
        next_state, r, term, _, _ = self.env.step(a)
        sprime = np.ravel_multi_index(next_state, self.env_shape)
        self.Q[(sidx, a)] = self.calc_non_dominated(sidx)
        self.R[(sidx, a)] += (r - self.R[(sidx, a)]) / self.Nsa[(sidx, a)]
        self.Nsa[(sidx, a)] += 1
        self.Ns[sidx] += 1
        next_state = torch.from_numpy(np.array([sprime], dtype=np.float32)).to(device)
        self._simulate(next_state, d + 1, term)


    def _rollout(self, s, term, sidx):
        if term:
            # need to return a vector of rewards as this is a multi-objective problem
            return np.array([0.] * self.num_objectives)
        else:
            # we are using a NN set just return V here
            return self._computeQ(s, sidx) 

    def _computeQ(self, state, sidx):
        # The value can be derived from the Qvalues in the initial state
        # by calling the neural network
        #state = torch.from_numpy(np.tile(np.array([state]), (self.num_objectives, 1))).to(device)
        with torch.no_grad():
            q_values = np.array([tuple(self.Qnet[k](state).detach().cpu().numpy()) for k in range(self.num_objectives)])
        #q_values = q_values.cpu().numpy()
        #Vs = torch.max(q_values, 1)
        for a in range(self.num_actions):
            self.Q[(sidx, a)] = self.Q[(sidx, a)].union({tuple(q_values[:, a])})
            self.Q[(sidx, a)] = self.calc_non_dominated(sidx)


    #def track_policy(self, vec):
    #    """Track a policy from its return vector.
    #    Args:
    #        vec (array_like): The return vector to track.
    #    """
    #    target = np.array(vec)
    #    state, _ = self.env.reset()
    #    terminated = False
    #    truncated = False
    #    total_rew = np.zeros(self.num_objectives)
    #    #
    #    while not (terminated or truncated):
    #        state = np.ravel_multi_index(state, self.env_shape)
    #        state_ = torch.from_numpy(np.repeat([state], self.num_objectives))
    #        new_target = False
    #        #
    #        for action in range(self.num_actions):
    #            im_rew = self.avg_reward[state, action]
    #            non_dominated_set = self.non_dominated[state][action]
    #            for q in non_dominated_set:
    #                q = np.array(q)
    #                if np.all(self.gamma * q + im_rew == target):
    #                    state, reward, terminated, truncated, _ = self.env.step(action)
    #                    total_rew += reward
    #                    target = q
    #                    new_target = True
    #                    break
    #                    #
    #            if new_target:
    #                break
    #    #                
    #    return total_rew

    def get_local_pcs(self, state: int = 0):
        """Collect the local PCS in a given state.
        Args:
            state (int): The state to get a local PCS for. (Default value = 0)
        Returns:
            Set: A set of Pareto optimal vectors.
        """
        if state in self.Tree:
            q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
            candidates = set().union(*q_sets)
            return get_non_dominated(candidates)
