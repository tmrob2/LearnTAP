import numpy as np
from collections import defaultdict
from typing import List, Tuple
from utils.dfa import DFA
import torch
import random

class MAMOMCTSNode:
    def __init__(self, 
        state, 
        num_actions,
        model, 
        tasks: List[DFA], state_shape: Tuple,
        ref_point, seed, f, h, 
        non_dominated_ref, avg_rewards_ref, counts,
        parent=None, parent_action=None, reward=None, p=1.):
        #
        self.state = state
        self.num_actions = num_actions
        self.id = id
        self.parent = parent
        self.parent_action = parent_action
        self.children: List[MAMOMCTSNode] = []
        self._num_of_visits = 0
        self._results = defaultdict(int)
        self.model = model
        self.tasks = tasks
        self.counts = counts
        self.non_dominated = non_dominated_ref
        self.avg_rewards = avg_rewards_ref
        self.state_shape = state_shape
        self.ref_point = ref_point
        self._untried_actions = self.untried_actions()
        #self.rng = np.random.default_rng(seed)
        self.f = f # scalar function to evalue Q set 
        self.calc_non_dominated = h
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.epsilon_decay = 0.99
        self.final_epsilon = 0.1
        self.p = p

    def untried_actions(self):
        return self.enabled_actions(self.state)
    
    def enabled_actions(self, state):
        valid_actions = []
        actions = list(range(4))
        for action in actions:
            next_state = np.array(state[1:3]) * 10. + self.model.env.dir[action]
            if self.model.env.is_valid_state(np.round(next_state, 1)):
                valid_actions.append(action)        
        Q = state[3:]
        if all([t.model_is_idle(Q[j]) for j, t in enumerate(self.tasks)]):
            valid_actions.append(4)
        for j, t in enumerate(self.tasks):
            if t.model_not_started(Q[j]):
                valid_actions.append(6 + j)
        return valid_actions
    
    def update_q(self):
        pass

    def n(self):
        return self._num_of_visits

    def make_table_index(self, state):
        env_coors = (state[1:3] * (np.array(self.state_shape)-1.)[1:3]).astype(np.int32)
        state_int = [int(state[0]), env_coors[0], env_coors[1]]
        state_int.extend(map(lambda x: int(x), list(state[3:])))
        state_int = tuple(state_int)
        sidx = np.ravel_multi_index(state_int, self.state_shape)
        return sidx

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
        next_state, reward, p = self.model(self.state, action, self.p)
        
        #print("state", self.state, "next state:", next_state)
        child_node = MAMOMCTSNode(
            next_state, 
            num_actions=self.num_actions,
            model=self.model,
            state_shape=self.state_shape,
            ref_point=self.ref_point,
            seed=1234,
            f=self.f,
            h=self.calc_non_dominated,
            counts=self.counts,
            parent=self, 
            parent_action=action,
            non_dominated_ref=self.non_dominated,
            avg_rewards_ref=self.avg_rewards,
            tasks=self.tasks,
            p=p
        )
        #print("child added ", child_node.state, "action", child_node.parent_action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        Q = list(self.state[3:])
        return all([t.model_is_finished(Q[i]) for i, t in enumerate(self.tasks)])

    def terminal_rollout(self, state):
        Q = list(state[3:])
        return all([t.model_is_finished(Q[i]) for i, t in enumerate(self.tasks)])

    def rollout(self):
        # need to think about this a little bit more because
        # we may need to update the Qsets and Ravg values in each rollout
        current_rollout_state = self.state
        p = self.p
        steps = 0
        while not self.terminal_rollout(current_rollout_state):
            actions = self.enabled_actions(current_rollout_state)
            #action = np.random.randint(len(actions))
            action = random.choice(actions)

            # We use our world model to simulate the future
            with torch.no_grad():
                next_state, reward, p = self.model(current_rollout_state, action, p)
            # we will need to add the ground truth data to the 
            # dataset here fore supervised learning
            # convert the next state into an index
            state = self.make_table_index(current_rollout_state)
            next_state_idx = self.make_table_index(next_state)
            #count_mask = np.array([0 if t.is_finished() else 1 for t in self.tasks])
            self.counts[state, action] += 1
            self.non_dominated[state][action] = self.calc_non_dominated(int(next_state_idx))
            self.avg_rewards[state, action] += (reward - self.avg_rewards[state, action]) / self.counts[state, action]
            #print("state", state, "act", action, "ND", self.non_dominated[state][action])
            
            #print(f"state", s, "action", action, "next_state", 
            #    next_state, "Q", Q, 
            #    "value: ", self.model.env.get_map_value((int(s[1] * 10), int(s[2] * 10))),
            #    "p", p
            #)
            #print("counts", self.counts[state][action])
            #print("R", self.avg_rewards[state, action])
            #print("s: ", current_rollout_state, "act", action, "s'", next_state)
            current_rollout_state = next_state
            steps += 1
            #print("rollout of state", self.state, "lasted ", steps)
        return p


    def backpropagate(self):
        # actually compute the scores
        # In this step all the statisics for the nodes are updated
        # Until the planned node is reached, the number of visits
        # for each node is incremented by 1. 
        self._num_of_visits += 1 # i.e. path length in the root node
        #if self.parent:
        #    #print("state", self.parent.state, "child_state", self.state, "action", self.parent_action)
        #    # TODO back propagate with the real environment or the model?
        #    # We also use the model to back propagate the action chosen
        #    state_idx = self.make_table_index(self.parent.state)
        #    self.counts[state_idx, self.parent_action] += 1
        #    self.non_dominated[state_idx][self.parent_action] = \
        #        self.calc_non_dominated(self.make_table_index(self.state))
        #    self.avg_rewards[state_idx, self.parent_action] += \
        #        (1 - self.avg_rewards[state_idx, self.parent_action]) \
        #        / self.counts[state_idx, self.parent_action]
        #    # Accumulate some results
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

        real_state = np.array([self.state[0]] + list(self.state[1:3] * (np.array(self.state_shape[1:3]) - 1)) + list(self.state[3:]))
        #print("best child real state", real_state.astype(np.int32))
        state = np.ravel_multi_index(real_state.astype(np.int32), self.state_shape)
        hypervolume = self.f(state)
        for (i, c) in enumerate(self.children):
            w = hypervolume[i] / c.n() + c_param * np.sqrt((2 * np.log(self.n())/ c.n()))
            weights.append(w)
        best_child_index = np.random.choice(np.argwhere(weights==np.max(weights)).flatten())
        #print("best next state: ", self.children[best_child_index].state)
        return self.children[best_child_index]

    def _tree_policy(self):
        # selects node to run rollout on
        #print("current state", self.state)
        current_node = self
        while not current_node.is_terminal_node():
            #print("current node", current_node.state)
            if not current_node.is_fully_expanded():
                #print("expand")
                return current_node.expand()
            else:
                #print("already expanded; get best child")
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 10
        for i in range(simulation_no):
            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
            v = self._tree_policy()
            p = v.rollout()
            v.backpropagate()
        # notice here we return the best child with c_param as 0
        return self.best_child(c_param=0.1).parent_action
            

        

    

