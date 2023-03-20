import numpy as np
import sim
import utils
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


OBS_CENTER = np.array([[0.9, 0], [0.25, 0.5], [-0.3, 0.5], [-1, 0.1], [0.3, -0.8]])
OBS_RADIUS = np.array([0.5, 0.3, 0.2, 0.5, 0.4])
SPH_RADIUS = 0.02

FK_Solver = utils.FKSolver() # forward kinematics solver


########## Task 1: Particle Weights Calculation ##########

def dist_to_closest_obs(x, y):
    """
    Find the distance between the stick's sphere end centered at (x, y) 
    to its closest cylinder obstacle.
    args:       x: The x-coordinate of the spherical end.
                y: The y-coordinate of the spherical end. 
    returns: dist: The distance to the closest obstacle.
    """
    ########## TODO ##########
    dist = 0.0


    ##########################
    return dist

def cal_weights(particles, obv, sigma=0.05):
    """
    Calculate the weights for particles based on the given observation.
    args:  particles: The particles represented by their states
                      Type: numpy.ndarray of shape (# of particles, 3)
                 obv: The given observation (the robot's joint angles). 
                      Type: numpy.ndarray of shape (7,)
               sigma: The standard deviation of the Gaussian distribution
                      for calculating likelihood (default: 0.05).
    returns: weights: The weights of all particles.
                      Type: numpy.ndarray of shape (# of particles,)
    """
    ########## TODO ##########
    weights = None


    ##########################
    return weights

def most_likely_particle(particles, obv):
    """
    Find the most likely particle.
    args:  particles: The particles represented by their states
                      Type: numpy.ndarray of shape (# of particles, 3)
                 obv: The given observation (the robot's joint angles). 
                      Type: numpy.ndarray of shape (7,)
    returns:     idx: The index of the most likely particle
                      Type: int
    """
    ########## TODO ##########
    idx = 0


    ##########################
    return idx


########## Task 2: Particle Filter ##########

def particle_filter(panda_sim, obvs, num_particles, sigma=0.05):
    """
    The Particle Filtering algorithm. 
    args:    panda_sim: The instance of the simulation. 
                        Type: sim.PandaSim (provided)
                  obvs: The given observations (the robot's joint angles)
                        Type: numpy.ndarray of shape (# of observations, 7)
         num_particles: The number of particles. 
                        Type: int
                 sigma: The standard deviation of the Gaussian distribution
                        for calculating likelihood (default: 0.05).
    returns:       est: The estimate of the pose of the robot base. 
                        Type: numpy.ndarray of shape (3,)
    """
    # initialize the particles and weights 
    particles = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(num_particles, 3))
    weights = np.ones(shape=(num_particles,)) / num_particles

    # configure the visualization
    ax = utils.config_plot_ax()
    utils.plot_pf(ax, particles, panda_sim.loc)
    plt.pause(0.01)

    for obv in obvs:
        panda_sim.set_joint_values(obv)

        ########## TODO ##########
        


        ##########################

        # plot the particles in the visualization
        utils.plot_pf(ax, particles, panda_sim.loc)
        plt.pause(0.01)
    plt.show()
    est = particles.mean(0)
    return est


########## Task 3: Generate Observations Online ##########

def get_one_obv(panda_sim):
    """
    Control the robot in simulation to obtain an observation.
    args: panda_sim: The instance of the simulation. 
                     Type: sim.PandaSim (provided)
    returns:    obv: One observation found by this function.
                     Type: numpy.ndarray of shape (7,)
    """
    ########## TODO ##########
    obv = None
    


    ##########################
    return obv

def particle_filter_online(panda_sim, num_particles, sigma=0.05):
    """
    The online Particle Filtering algorithm. 
    args:     panda_sim: The instance of the simulation. 
                         Type: sim.PandaSim (provided)
          num_particles: The number of particles. 
                         Type: int
                  sigma: The standard deviation of the Gaussian distribution
                         for calculating likelihood (default: 0.05).
    returns:        est: The estimate of the pose of the robot base. 
                         Type: numpy.ndarray of shape (3,)
    """
    # # initialize the particles and weights 
    particles = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(num_particles, 3))
    weights = np.ones(shape=(num_particles,)) / num_particles

    # configure the visualization
    ax = utils.config_plot_ax()
    utils.plot_pf(ax, particles, panda_sim.loc)
    plt.pause(0.01)

    ########## TODO ##########
    num_iters = 100 # The number of iterations. Feel free to change the value
    for _ in range(num_iters):
        


        

        # plot the particles in the visualization 
        utils.plot_pf(ax, particles, panda_sim.loc)
        plt.pause(0.01)
    ##########################
    est = particles.mean(0)
    return est
