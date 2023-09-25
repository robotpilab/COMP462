import numpy as np


class StateSampler(object):
    """
    The state sampler of Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim = self.pdef.get_state_dimension() # dimensionality of state space
        self.low = self.pdef.bounds_state.low # the lower bounds of the state space
        self.high = self.pdef.bounds_state.high # the upper bounds of the state space

    def sample(self):
        """
        Uniformly random sample a state vector.
        returns: stateVec: The sampled state vector.
                           Type: numpy.ndarray of shape (self.dim,)
        """
        stateVec = np.random.uniform(self.low, self.high, self.dim)
        return stateVec


class ControlSampler(object):
    """
    The control sampler for Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim = self.pdef.get_control_dimension() # dimensionality of the control space
        self.low = self.pdef.bounds_ctrl.low # the lower bounds of the control space
        self.high = self.pdef.bounds_ctrl.high # the upper bounds of the control space

    def sample_to(self, nnode, rstateVec, k):
        """
        Sample k candidates controls from nnode and return the control 
        whose outcome state is nearest to rstateVec.
        args:     nnode: The node from where the controls are sampled.
                         Type: rrt.Node
              rstateVec: The reference state vector towards which the controls are sampled.
                         Type: numpy.ndarray
                      k: The number of candidates controls.
                         Type: int
        returns:  bctrl: The best control which leads to a state nearest to rstateVec.
                         Type: numpy.ndarray of shape (self.dim,)
                               or None if all the k candidate controls lead to an invalid state
                 ostate: The outcome state of the best control.
                         Type: dict, {"stateID": int, "stateVec": numpy.ndarray}
                               or None if bctrl is None
        """
        nstate = nnode.state
        assert k >= 1
        controls = []
        pstates = []
        dists = []
        i = 0
        while (i < k):
            ctrl = np.random.uniform(self.low, self.high, self.dim)
            pstate, valid = self.pdef.propagate(nstate, ctrl)
            if valid and self.pdef.is_state_valid(pstate):
                dist = self.pdef.distance_func(pstate["stateVec"], rstateVec)
                controls.append(ctrl)
                pstates.append(pstate)
                dists.append(dist)
            i += 1

        bctrl, ostate = None, None 
        if len(dists) > 0:
            best_i = np.argmin(dists)
            bctrl, ostate = controls[best_i], pstates[best_i]
        return bctrl, ostate
        