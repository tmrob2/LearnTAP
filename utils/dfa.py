class Progress:
    NOT_STARTED = 0
    IN_PROGRESS = 1
    JUST_FINISHED = 2
    FINISHED = 3
    FAILED = 4

class DFA:
    def __init__(self, init, acc, rej, jfin) -> None:
        self.handlers = {}
        self.start_state = init
        self.current_state = self.start_state
        self.acc = acc
        self.rej = rej
        self.jfin = jfin
        self.states = []
        self.progress_flag = Progress.NOT_STARTED

    def add_state(self, name, f):
        self.states.append(name)
        self.handlers[name] = f
    
    def next(self, state, data, agent):
        if state is not None:
            f = self.handlers[state]
            new_state = f(state, data, agent)
            if data["update"]:
                self.current_state = new_state
            return new_state
        else:
            return state
    
    def reset(self):
        self.current_state = self.start_state
        self.progress_flag = Progress.NOT_STARTED
    
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

    def in_progress(self):
        fin = self.acc + self.rej
        if self.current_state >= 1 and self.current_state not in fin:
            return True
        else:
            return False

    def model_in_progress(self, q):
        fin = self.acc + self.rej
        if q >= 1 and self.current_state not in fin:
            return True
        else:
            return False

    def set_current_state(self, q):
        self.current_state = q

    def model_rewards(self, q, value):
        if q in self.jfin:
            return value
        else:
            return 0

    def is_finished(self):
        fin = self.acc + self.rej
        if self.current_state in fin:
            return True
        else:
            return False

    def is_complete(self):
        fin = self.acc + self.rej + self.jfin
        if self.current_state in fin:
            return True
        else: 
            return False

    def model_is_complete(self, q):
        fin = self.acc + self.rej + self.jfin
        if q in fin:
            return True
        else:
            return False

    def model_is_finished(self, q):
        fin = self.acc + self.rej
        if q in fin:
            return True
        else:
            return False

    def is_idle(self):
        idle = self.acc + self.rej + [self.start_state]
        if self.current_state in idle:
            return True
        else:
            return False

    def model_is_idle(self, q):
        idle = self.acc + self.rej + [self.start_state]
        if q in idle:
            return True
        else:
            return False

    def model_not_started(self, q):
        if q == self.start_state:
            return True
        else:
            return False

    def is_succ(self):
        if self.current_state in self.acc:
            return True
        else:
            return False

    def model_is_succ(self, q):
        if q in self.acc:
            return True
        else:
            return False
