import numpy as np
import sim


class Bounds(object):
    """
    Bounds of the state space for a motion planning problem.
    """
    
    def __init__(self, dim):
        """
        args: dim: The dimensionality of the state space.
                   Type: int
        """
        self.dim = dim
        # default bounds: [-1, 1] for all dimensions
        self.low = -np.ones(shape=(self.dim,))
        self.high = np.ones(shape=(self.dim,))

    def set_bounds(self, i, low, high):
        """
        Set bounds for one dimension of the state space.
        args:    i: The index of the dimension to set.
                    Type: int
               low: The lower bound.
              high: The upper bound.   
        """
        assert i >= 0 and i < self.dim
        self.low[i] = low
        self.high[i] = high

    def is_satisfied(self, state):
        stateVec = state["stateVec"]
        if not np.all(stateVec >= self.low):
            return False
        if not np.all(stateVec <= self.high):
            return False
        return True


class ProblemDefinition(object):
    """
    The definition of the motion planning problem to be solved.
    This includes necessary information and specifications of the problem 
    (e.g., start state, goal, bounds, etc).
    """

    def __init__(self, panda_sim):
        """
        args: panda_sim: The simulation environment of the Franka Panda robot based on pybullet.
                         Type: sim.PandaSim
        """
        self.panda_sim = panda_sim
        self.start_state = self.panda_sim.save_state()
        self.goal = None
        self.dim_state = sim.pandaNumDofs + 3 * self.panda_sim.num_objects # dimensionality of the state space
        self.dim_ctrl = 4 # dimensionality of the control space
        self.bounds_state = Bounds(self.dim_state) # bounds of the state space
        self.bounds_ctrl = Bounds(self.dim_ctrl) # bounds of the control space

    def get_state_dimension(self):
        return self.dim_state

    def get_control_dimension(self):
        return self.dim_ctrl

    def distance_func(self, stateVec1, stateVec2):
        """
        Calculate the Euclidean distance of two state vector(s).
        """
        dim = stateVec1.shape[-1]
        dists = np.linalg.norm(stateVec1.reshape(-1, dim) - stateVec2.reshape(-1, dim), axis=1)
        return dists

    def get_goal(self):
        return self.goal

    def set_goal(self, goal):
        self.goal = goal

    def get_start_state(self):
        return self.start_state
    
    def set_start_state(self, state_st):
        self.start_state = state_st

    def set_state_bounds(self, bounds):
        self.bounds_state = bounds

    def set_control_bounds(self, bounds):
        self.bounds_ctrl = bounds

    def is_state_valid(self, state):
        """
        Check if a state is valid or not.
        args: state: The query state of the system.
                     Type: dict, {"stateID": int, 
                                  "stateVec": numpy.ndarray of shape (self.dim_state,)}
        returns: Ture or False.
        """
        ########## TODO ##########
        

        ##########################
        
    def propagate(self, nstate, control):
        self.panda_sim.restore_state(nstate)
        self.panda_sim.execute(control)
        state = self.panda_sim.save_state()
        return state
