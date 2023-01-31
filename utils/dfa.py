class Progress:
    NOT_STARTED = 0
    IN_PROGRESS = 1
    JUST_FINISHED = 2
    FINISHED = 3
    FAILED = 4

class DFA:
    def __init__(self, init, acc, rej) -> None:
        self.handlers = {}
        self.start_state = init
        self.current_state = self.start_state
        self.acc = acc
        self.rej = rej
        self.states = []
        self.progress_flag = Progress.NOT_STARTED

    def add_state(self, name, f):
        self.states.append(name)
        self.handlers[name] = f
    
    def next(self, state, data, agent):
        if state is not None:
            f = self.handlers[state]
            new_state = f(self.current_state, data, agent)
            if self.progress_flag >= Progress.IN_PROGRESS:
                self.update_progress(new_state)
                self.current_state = new_state
        else:
            return state
    
    def reset(self):
        self.current_state = self.start_state
        self.progress_flag = Progress.NOT_STARTED

    def activate_task(self):
        if self.progress_flag == Progress.NOT_STARTED:
            self.progress_flag = Progress.IN_PROGRESS
    
    def update_progress(self, state):
        if state in self.acc and self.progress_flag >= Progress.IN_PROGRESS:
            if self.progress_flag < 2:
                self.progress_flag = Progress.JUST_FINISHED
            else:
                self.progress_flag = Progress.FINISHED
        elif state in self.rej:
            self.progress_flag = Progress.FAILED
    
    def assign_reward(self, one_off_reward):
        if self.progress_flag == Progress.JUST_FINISHED:
            return one_off_reward
        else:
            return 0.