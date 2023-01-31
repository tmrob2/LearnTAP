import numpy as np

def epsilon_greedy(q, epsilon):
  """Epsilon-greedy policy: selects the maximum value action with probabilty
  (1-epsilon) and selects randomly with epsilon probability.

  Args:
    q (ndarray): an array of action values
    epsilon (float): probability of selecting an action randomly

  Returns:
    int: the chosen action
  """
  be_greedy = np.random.random() > epsilon
  if be_greedy:
    action = np.argmax(q)
  else:
    action = np.random.choice(len(q))

  return action

class world(object):
    def __init__(self):
        return

    def get_outcome(self):
        print("Abstract method, not implemented")
        return

    def get_all_outcomes(self):
        outcomes = {}
        for state in range(self.n_states):
            for action in range(self.n_actions):
                next_state, reward = self.get_outcome(state, action)
                outcomes[state, action] = [(1, next_state, reward)]
        return outcomes

class QuentinsWorld(world):
    """
    World: Quentin's world.
    100 states (10-by-10 grid world).
    The mapping from state to the grid is as follows:
    90 ...       99
    ...
    40 ...       49
    30 ...       39
    20 21 22 ... 29
    10 11 12 ... 19
    0  1  2  ...  9
    54 is the start state.
    Actions 0, 1, 2, 3 correspond to right, up, left, down.
    Moving anywhere from state 99 (goal state) will end the session.
    Landing in red states incurs a reward of -1.
    Landing in the goal state (99) gets a reward of 1.
    Going towards the border when already at the border will stay in the same
        place.
    """
    def __init__(self):
        self.name = "QuentinsWorld"
        self.n_states = 100
        self.n_actions = 4
        self.dim_x = 10
        self.dim_y = 10
        self.init_state = 54
        self.shortcut_state = 64

    def toggle_shortcut(self):
      if self.shortcut_state == 64:
        self.shortcut_state = 2
      else:
        self.shortcut_state = 64

    def get_outcome(self, state, action):
        if state == 99:  # goal state
            reward = 0
            next_state = None
            return next_state, reward
        reward = 0  # default reward value
        if action == 0:  # move right
            next_state = state + 1
            if state == 98:  # next state is goal state
                reward = 1
            elif state % 10 == 9:  # right border
                next_state = state
            elif state in [11, 21, 31, 41, 51, 61, 71,
                           12, 72,
                           73,
                           14, 74,
                           15, 25, 35, 45, 55, 65, 75]:  # next state is red
                reward = -1
        elif action == 1:  # move up
            next_state = state + 10
            if state == 89:  # next state is goal state
                reward = 1
            if state >= 90:  # top border
                next_state = state
            elif state in [2, 12, 22, 32, 42, 52, 62,
                           3, 63,
                           self.shortcut_state,
                           5, 65,
                           6, 16, 26, 36, 46, 56, 66]:  # next state is red
                reward = -1
        elif action == 2:  # move left
            next_state = state - 1
            if state % 10 == 0:  # left border
                next_state = state
            elif state in [17, 27, 37, 47, 57, 67, 77,
                           16, 76,
                           75,
                           14, 74,
                           13, 23, 33, 43, 53, 63, 73]:  # next state is red
                reward = -1
        elif action == 3:  # move down
            next_state = state - 10
            if state <= 9:  # bottom border
                next_state = state
            elif state in [22, 32, 42, 52, 62, 72, 82,
                           23, 83,
                           84,
                           25, 85,
                           26, 36, 46, 56, 66, 76, 86]:  # next state is red
                reward = -1
        else:
            print("Action must be between 0 and 3.")
            next_state = None
            reward = None
        return int(next_state) if next_state is not None else None, reward


def q_learning(state, action, reward, next_state, value, params):
  """Q-learning: updates the value function and returns it.

  Args:
    state (int): the current state identifier
    action (int): the action taken
    reward (float): the reward received
    next_state (int): the transitioned to state identifier
    value (ndarray): current value function of shape (n_states, n_actions)
    params (dict): a dictionary containing the default parameters

  Returns:
    ndarray: the updated value function of shape (n_states, n_actions)
  """
  # value of previous state-action pair
  prev_value = value[int(state), int(action)]

  # maximum Q-value at current state
  if next_state is None or np.isnan(next_state):
      max_value = 0
  else:
      max_value = np.max(value[int(next_state)])

  # reward prediction error
  delta = reward + params['gamma'] * max_value - prev_value

  # update value of previous state-action pair
  value[int(state), int(action)] = prev_value + params['alpha'] * delta

  return value


def learn_environment(env: QuentinsWorld, model_updater, planner, params, max_steps,
                      n_episodes, shortcut_episodes=None):
    # start with a uniform value function
    value = np.ones((env.n_states, env.n_actions))

    # Run learning
    reward_sums = np.zeros(n_episodes)
    episode_steps = np.zeros(n_episodes)

    # Dyna-Q state
    model = np.nan * np.zeros((env.n_states, env.n_actions, 2))

    # loop over episodes
    for episode in range(n_episodes):
        if shortcut_episodes is not None and episode == shortcut_episodes:
            env.toggle_shortcut()
            state = 64
            action = 1
            next_state, reward = env.get_outcome(state, action)
            model[state, action] = reward, next_state
            value = q_learning(state, action, reward, next_state, value, params)

        state = env.init_state
        reward_sum = 0

        for t in range(max_steps):
            # choose the next action
            action = epsilon_greedy(value[state], params['epsilon'])

            # observe the outcome of the action on the env
            next_state, reward = env.get_outcome(state, action)

            # sum rewards obtained
            reward_sum += reward

            # update the model
            model = model_updater(model, state, action, reward, next_state)

            # execute planner
            value = planner(model, value, params)

            if next_state is None:
                break
            state = next_state

        reward_sums[episode] = reward_sum
        episode_steps[episode] = t + 1

    return value, reward_sums, episode_steps


def dyna_q_model_update(model, state, action, reward, next_state):
  """ Dyna-Q model update

  Args:
    model (ndarray): An array of shape (n_states, n_actions, 2) that represents
                     the model of the world i.e. what reward and next state do
                     we expect from taking an action in a state.
    state (int): the current state identifier
    action (int): the action taken
    reward (float): the reward received
    next_state (int): the transitioned to state identifier

  Returns:
    ndarray: the updated model
  """
  ###############################################################
  ## A model is an ndparray (nstates, nactions, 2)
  # When we query a state, action we want as return a reward and
  # next state, therefore how model should have a tuple (r, s')
  
  ###############################################################

  # Update our model with the observed reward and next state
  model[state, action] = reward, next_state # suffices

  return model


def dyna_q_planning(model, value, params):
  """ Dyna-Q planning

  Args:
    model (ndarray): An array of shape (n_states, n_actions, 2) that represents
                     the model of the world i.e. what reward and next state do
                     we expect from taking an action in a state.
    value (ndarray): current value function of shape (n_states, n_actions)
    params (dict): a dictionary containing learning parameters

  Returns:
    ndarray: the updated value function of shape (n_states, n_actions)
  """
  ############################################################
  
  #############################################################
  # Perform k additional updates at random (planning)
  for _ in range(params['k']):
    # Find state-action combinations for which we've experienced a reward i.e.
    # the reward value is not NaN. The outcome of this expression is an Nx2
    # matrix, where each row is a state and action value, respectively.
    candidates = np.array(np.where(~np.isnan(model[:,:,0]))).T

    # Write an expression for selecting a random row index from our candidates
    idx = np.random.randint(0, candidates.shape[0], 1)[0]

    # Obtain the randomly selected state and action values from the candidates
    state, action = candidates[idx]

    # Obtain the expected reward and next state from the model
    reward, next_state = model[state, action]

    # Update the value function using Q-learning
    value = q_learning(state, action, reward, next_state, value, params)

  return value