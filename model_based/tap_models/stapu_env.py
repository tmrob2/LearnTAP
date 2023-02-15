import gym
from gym.spaces import Box, Discrete
import numpy as np
from typing import List
from model_based.ma_envs.agent import Agent
from utils.dfa import DFA, Progress
import pygame
from model_based.ma_envs.multi_agent_deep_sea_treasure.ma_deep_sea_treasure import DeepSeaTreasure

class STAPU(gym.Env):
    """
    A simultaneous task allocation and planning under uncertainty
    environment which handles multi-agent task allocation and 
    planning.

    Takes a multi-agent gymnasium enviroment as
    well as a number of tasks, the tasks can be run-time 
    verification LTL
    """

    def __init__(self, base_env: gym.Env, tasks: List[DFA] = [], float_state = False, env_shape = None):
        """
        base_env must be a multi-agent environment otherwise will fail
        """
        # The number of agents is included in the base environment
        self.tasks = tasks
        self.env = base_env
        self.env.action_space = Discrete(6 + len(tasks))
        #self.env.action_space = Discrete(4)
        low_bound = self.env.observation_space.low
        high_bound = self.env.observation_space.high
        self.env_shape = (high_bound[0] - low_bound[0] + 1, high_bound[1] - low_bound[1] + 1) if env_shape is None else env_shape
        self.state_shape = [self.env.num_agents, self.env_shape[0], self.env_shape[1]]
        self.state_shape.extend([max(task.states) + 1 for task in self.tasks])
        self.float_state = float_state
        obs_type = np.float32 if self.float_state else np.int32
        self.reward_norm = np.array([1.] * (self.env.num_agents + len(tasks)))
        self.p = 1.0
        if self.float_state:
            self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=obs_type)
        else:
            low = [0]
            low.extend(base_env.observation_space.low)
            low.extend([min(task.states) for task in tasks])
            high = [base_env.num_agents]
            high.extend(base_env.observation_space.high)
            high.extend([max(task.states) for task in tasks])
            state_ref = 1 + 2 + len(tasks)
            self.observation_space = Box(low=np.array(low), high=np.array(high), shape=(state_ref,), dtype=obs_type)

        num_actions = 6 + len(tasks)
        #num_actions = 4
        self.action_space = Discrete(num_actions)
        self.reward_space = Box(
            low=np.array([0.] * self.env.num_agents + [0.] * len(tasks)),
            high=np.array([-1] * self.env.num_agents + [np.max(self.env.sea_map)] * len(tasks)),
            dtype=np.float32,
        )

    def step(self, action):
        # only one of the agents should ever be active at any given time
        # The state looks like (active_agent, current_agent_state, q1, ..., qm)
        #
        # If the action is one of the special switch actions then an agent
        # will hand over control to the next agent unless it is the last agent
        #
        # Let act = 6 be the action to do nothing
        # Let act = 5 be the switch transition handing over the task to
        # Let act > 6 be the signal to activate a task
        # then next agent
        done = False
        active_agent = list(filter(lambda x: x.active, self.env.agents))[0].agent_idx
        actions = [5] * self.env.num_agents
        actions[active_agent] = action
        # check if the agent has a task active, if it does not then it should not be
        # able to interact with the environment, otherwise an agent may infinitely
        # hold tasks from other agents
        Q = [t.current_state for t in self.tasks]
        x, y = tuple(self.env.get_state(active_agent))
        value = self.env.get_map_value((int(x), int(y)))
        inProgress = any([t.in_progress() for t in self.tasks])
        if action < 4 and inProgress:
            R_ = [0.] * self.env.num_agents
            R_[active_agent] =  -1. if value >= 0 else value
        else:
            R_  = [0.] * (self.env.num_agents + len(self.tasks))
            R_[active_agent] =  0 if value >= 0 else value
        R_[self.env.num_agents:] = [
            self.tasks[j].model_rewards(Q[j], self.p)
            for j in range(len(self.tasks))
        ]
        if inProgress and action < 4:
            states, _, _, _, _ = self.env.step(actions)
        else:
            states = [agent.current_state for agent in self.env.agents]
        #agent_task_activated_reward = 0
        # term and trunc have no effect and only the global DFA will 
        # dictate whether the episode should end or not
        # Each agent will have an updated state
        # but we only need the updated state of the active agent
        #state = self.env.get_state(active_agent.agent_idx)
        #istate = np.ravel_multi_index(states[active_agent], self.env_shape)
        # Then we also need to report on the task progress as well
        Q = []
        #print("Q", [t.current_state for t in self.tasks])
        if self.float_state:
            x, y = tuple((self.env.get_state(active_agent) * (np.array(self.env_shape) - 1)).astype(np.int32))
        else:
            x, y = tuple(self.env.get_state(active_agent))
        if self.env.get_map_value((int(x), int(y))) < 0:
            self.p -= 0.2
        data = {"env": self.env, "x": int(x), "y": int(y), "activate": False, "update": True, "p": self.p}
        for j in range(len(self.tasks)):
            # compute the task progress
            if action == j + 6:
                data["activate"] = True
            else:
                data["activate"] = False
            qprime = self.tasks[j].next(self.tasks[j].current_state, data, active_agent)
            Q.append(qprime)

        #if any(q > 1 for q in Q):
        #    print("Q", Q)
        
        # if all of the tasks that an agent is undertaking are either not
        # started or finished then it can enable the special action 
        # otherwise the special action has no effect
        if action == 4 and \
            all([t.is_idle() for t in self.tasks]) and \
            not all([t.is_finished() for t in self.tasks]) and \
            active_agent < self.env.num_agents - 1:
            # hand over the task to the next agent
            self.env.agents[active_agent].active = False
            self.env.agents[active_agent + 1].active = True
            active_agent += 1
        stapu_state = [active_agent]
        if self.float_state:
            stapu_state[0] = float(1 / self.env.num_agents)
        stapu_state.extend(list(states[active_agent]))
        stapu_state.extend(Q)
        if all([t.is_finished() for t in self.tasks]):
            done = True
        # truncated is always false
        #R = [np.sum(R_[:self.env.num_agents])] + R_[self.env.num_agents:] 
        return np.array(stapu_state), R_, done, False, {}

    def reset(self):
        state, _ = self.env.reset()
        active_agent = list(filter(lambda x: x.active, self.env.agents))[0].agent_idx
        for task in self.tasks:
            task.reset()
        Q = [task.current_state for task in self.tasks]
        state_ = [active_agent]
        state_.extend(state[active_agent].tolist())
        state_.extend(Q)
        self.p = 1.
        return np.array(state_), {}
        

    def render(self):
        self.env.render()


# define a simple test task to make sure that this is working
# A DFA is defined with tranisition functions 
# Informally, let the task be find two pieces of treasure
Qstates = [0, 1, 2] # 0 is initial, 1 is found treasure, 2 is fail

def treasure_found(q, env, agent):
    treasure_value = env.get_map_value(env.agents[agent].current_state)
    if treasure_value == 0:
        return q
    elif treasure_value > 0:
        return 1
    else:
        return 2

# Define the sink states
def accepting(q, env, agent):
    return 1

def failure(q, env, agent):
    return 2

def find_a_treasure():
    dfa = DFA(0, [1], [2], [])
    dfa.add_state(0, treasure_found)
    dfa.add_state(1, accepting)
    dfa.add_state(2, failure)
    return dfa

tasks = [find_a_treasure()]

if __name__ == "__main__":
    pygame.font.init()
    env = DeepSeaTreasure(render_mode="human")
    mamodel = STAPU(env, tasks)
    done = False
    mamodel.reset()
    while True:
        mamodel.render()
        # randomly generate an action for the active agent
        action = mamodel.env.action_space.sample()
        obs, r, done, _, _ = mamodel.step(action)
        active_agent = list(filter(lambda x: x.active, mamodel.env.agents))[0].agent_idx
        print(f"Active agent: {active_agent}, action: {action}, STAPU State: {obs} state idx: {np.ravel_multi_index(obs, mamodel.state_shape)}, reward: {r}, done: {done}, tasks active: {[task.progress_flag > 0 for task in mamodel.tasks]}, task status: {[task.progress_flag for task in mamodel.tasks]}")
        if done:
            print("Reset")
            mamodel.reset()



