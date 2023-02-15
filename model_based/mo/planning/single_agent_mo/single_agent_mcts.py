import numpy as np
from typing import List
from collections import defaultdict
import random
from morl_baselines.common.pareto import get_non_dominated


class SAMOModel:
    def __init__(self, env):
        self.env = env

    def __call__(self, state, action: int):
        # we want to call each of the agent models
        
        # first decompose the state, action pair
        #print("model input", state)
        x, y = state[0], state[1]
        # If the action is [0, 4) then perform an agent 
        # action in the environment
        next_state = np.array([x, y]) + self.env.dir[action]
        if self.env.is_valid_state(next_state):
            x, y = next_state[0], next_state[1]
        value = self.env.get_map_value((x, y))
        
        terminated = False
        if value == 0 or value < 0:
            value = 0
        else:
            terminated = True

        return np.array([x, y]), np.array([value, -1], dtype=np.float32), terminated 

class MOMCTSNode:
    def __init__(self, 
        state, 
        num_actions,
        model,
        ref_point, 
        seed, 
        f, 
        h, 
        non_dominated_ref, 
        avg_rewards_ref, 
        counts,
        parent=None, 
        parent_action=None
        ):
        #
        self.state = state
        self.num_actions = num_actions
        self.id = id
        self.parent = parent
        self.parent_action = parent_action
        self.children: List[MOMCTSNode] = []
        self._num_of_visits = 0
        self._results = defaultdict(int)
        self.model = model
        self.counts = counts
        self.non_dominated = non_dominated_ref
        self.avg_reward = avg_rewards_ref
        self.ref_point = ref_point
        self._untried_actions = self.untried_actions()
        #self.rng = np.random.default_rng(seed)
        self.f = f # scalar function to evalue Q set 
        self.calc_non_dominated = h
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.epsilon_decay = 0.99
        self.final_epsilon = 0.1
        self.state_shape = (11, 11)

    def untried_actions(self):
        return self.enabled_actions(self.state)

    def enabled_actions(self, state):
        valid_actions = []
        actions = list(range(4))
        for action in actions:
            next_state = state + self.model.env.dir[action]
            if self.model.env.is_valid_state(next_state):
                valid_actions.append(action)
        return valid_actions
        #return list(range(4))

    def update_q(self):
        pass

    def n(self):
        return self._num_of_visits

    def expand(self):
        action = self._untried_actions.pop() # remove an enabled action
        # a state will most likely be an np.array because we consider discrete
        # gym models in MTCS therefore we need to edit is to include
        # the action and create a tensor from it
        #assert isinstance(self.state, torch.Tensor)
        # convert the state to a tensor
        

        # create an action tensor
        #a_ = torch.tensor([float(action/self.num_actions)], dtype=torch.float32)
        #model_input_state = torch.cat((self.make_model_state(), a_), 0)

        # Get the next state from our world model
        #print("state", self.state)
        next_state, _, _ = self.model(self.state, action)
    
        #print("state", self.state, "next state:", next_state)
        child_node = MOMCTSNode(
            next_state, 
            num_actions=self.num_actions,
            model=self.model,
            ref_point=self.ref_point,
            seed=1234,
            f=self.f,
            h=self.calc_non_dominated,
            counts=self.counts,
            parent=self, 
            parent_action=action,
            non_dominated_ref=self.non_dominated,
            avg_rewards_ref=self.avg_reward
        )
        #print("child added ", child_node.state, "action", child_node.parent_action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        value = self.model.env.get_map_value(self.state)
        if value == 0 or value < 0:
            return False
        else:
            return True

    def terminal_rollout(self, state):
        value = self.model.env.get_map_value(state)
        if value < 0:
            return True
        else:
            return False

    def _make_table_index(self, state):
        sidx = np.ravel_multi_index(state, self.state_shape)
        return sidx

    def _get_q_set(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
        nd_array = np.array(list(self.non_dominated[state][action]))
        #print("state:", state, "action", action, nd_array, "r", self.avg_reward[state, action])
        q_array = self.avg_reward[state, action] + 1. * nd_array
        return {tuple(vec) for vec in q_array}

    def _calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state.

        Args:
            state (int): The current state.

        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        candidates = set().union(*[self._get_q_set(state, action) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated

    def rollout(self):
        current_rollout_state = self.state
        #steps = 0
        # check if child is terminal
        
        terminal = self.is_terminal_node()
        while not terminal:
            #print("state", current_rollout_state, "value", self.model.env.get_map_value((self.state[0], self.state[1])))
            actions = self.enabled_actions(current_rollout_state)
            #action = np.random.randint(len(actions))
            action = random.choice(actions)

            # We use our world model to simulate the future
            state = self._make_table_index(current_rollout_state)
            next_state, reward, terminal = self.model(current_rollout_state, action)
            # we will need to add the ground truth data to the 
            # dataset here fore supervised learning
            # convert the next state into an index
            next_state_idx = self._make_table_index(next_state)
            #count_mask = np.array([0 if t.is_finished() else 1 for t in self.tasks])
            self.counts[state, action] += 1
            self.non_dominated[state][action] = self._calc_non_dominated(next_state_idx)
            self.avg_reward[state, action] += (reward - self.avg_reward[state, action]) / self.counts[state, action]
            #if self.avg_rewards[state][action][1] > 30.:
            #print(f"state", current_rollout_state, 
            #    "action", action, 
            #    "next_state", next_state,
            #    "reward", reward,
            #    "r", self.avg_reward[state, action],
            #    "c", self.counts[state, action]
            #)
            current_rollout_state = next_state
            #steps += 1
        return

    def backpropagate(self):
        self._num_of_visits += 1
        if self.parent:
            self.parent.backpropagate()

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0


    def best_child(self, c_param=0.1):
        # we need to choose a child(action) based on the non
        # dominated action choice function
        # the actions scores are a scalar value because 
        # the hypervolume is a scalar function
        weights = []

        #real_state = np.array([self.state[0]] + list(self.state[1:3] * (np.array(self.state_shape[1:3]) - 1)) + list(self.state[3:]))
        #print("best child real state", real_state.astype(np.int32))
        state = self._make_table_index(self.state)
        hypervolume = self.f(state)
        for (i, c) in enumerate(self.children):
            w = hypervolume[i] / c.n() + c_param * np.sqrt((2 * np.log(self.n())/ c.n()))
            weights.append(w)
        best_child_index = np.random.choice(np.argwhere(weights==np.max(weights)).flatten())
        #print("best next state: ", self.children[best_child_index].state)
        return self.children[best_child_index]

    def _tree_policy(self):
        # selects node to run rollout on
        current_node = self
        while not current_node.is_terminal_node():
            #print("current node: ", current_node.state, 
            #    "value", self.model.env.get_map_value(current_node.state))
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                #print("already expanded; get best child")
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 1
        for _ in range(simulation_no):
            v = self._tree_policy()
            v.rollout()
            v.backpropagate()
        # notice here we return the best child with c_param as 0
        return self.best_child(c_param=0.1).parent_action