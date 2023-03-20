import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc


class FKSolver(object):
    """
    The Forward Kinematics solver for the 7-DoF Franka Panda robot.
    """

    def __init__(self):
        self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)
        self.bullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.panda = self.bullet_client.loadURDF("urdf/panda_stick.urdf", useFixedBase=True)

    def forward_kinematics_2d(self, joint_values):
        """
        Calculate the Forward Kinematics of the robot given joint angle values.
        args: joint_values: The joint angle values of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:         x: The x-coordinate of the stick's spherical end w.r.t. the robot base.
                         y: The y-coordinate of the stick's spherical end w.r.t. the robot base. 
        """
        for j in range(7):
            self.bullet_client.resetJointState(self.panda, j, joint_values[j])
        ee_state = self.bullet_client.getLinkState(self.panda, linkIndex=13)
        x, y, _ = ee_state[4]
        return x, y


def load_npy(file):
    with open(file, 'rb') as f:
        npy = np.load(f)
    return npy

def config_plot_ax():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    return ax

def plot_gt(ax, loc):
    ax.scatter(loc[0], loc[1], loc[2],
               c='r', s=200, alpha=1.0, label="groundtruth")

def plot_estimate(ax, est):
    ax.scatter(est[0], est[1], est[2],
               c='b', s=200, alpha=1.0, label="estimate")
               
def plot_particles(ax, particles):
    ax.scatter(particles[:, 0],
               particles[:, 1],
               particles[:, 2],
               c='g', s=50, alpha=0.5, label="particles")

def plot_pf(ax, particles, gt):
    ax.clear()
    plot_gt(ax, gt)
    plot_estimate(ax, particles.mean(0))
    plot_particles(ax, particles)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-np.pi, np.pi)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("theta")
    ax.legend()
