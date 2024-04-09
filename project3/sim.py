import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaFingerJoint1Index = 9
pandaFingerJoint2Index = 10
pandaNumDofs = 7

pandaJointRange = np.array([[-2.8973, 2.8973],
                            [-1.7628, 1.7628],
                            [-2.8973, 2.8973],
                            [-3.0718, -0.0698],
                            [-2.8973, 2.8973],
                            [-0.0175, 3.7525],
                            [-2.8973, 2.8973]])

pandaStartJoints = [0.0, -0.7853981633974483, 0.0, -2.356194490192345,
                    0.0, 1.5707963267948966, 0.7853981633974483]

SimTimeStep = 1./500.


class PandaSim(object):

  """
  The simulation environment of the 7-DoF Franka Panda robot based on pybullet.
  """

  def __init__(self, bullet_client, loc=[0.0, 0.0, 0.0]):
    self.bullet_client = bullet_client
    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    self.plane = self.bullet_client.loadURDF("plane.urdf")

    # setup the Panda robot arm
    quat = self.bullet_client.getQuaternionFromEuler([0.0, 0.0, loc[2]])
    self.panda = self.bullet_client.loadURDF("urdf/panda_stick.urdf", [loc[0], loc[1], 0.0], quat, useFixedBase=True, flags=flags)
    self.pandaNumJoints = self.bullet_client.getNumJoints(self.panda)
    print(self.pandaNumJoints)
    self.bullet_client.resetJointState(self.panda, pandaFingerJoint1Index, 0.01)
    self.bullet_client.resetJointState(self.panda, pandaFingerJoint2Index, 0.01)
    for j in range(pandaNumDofs):
      self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0,
                                        jointLowerLimit=pandaJointRange[j, 0], jointUpperLimit=pandaJointRange[j, 1])
      self.bullet_client.resetJointState(self.panda, j, pandaStartJoints[j])

    # spawn the obstacles
    self.env = self.bullet_client.loadURDF("urdf/env.urdf", [0.0, 0.0, 0.5], [0, 0, 0, 1], useFixedBase=True, flags=flags)

    self.loc = loc
    self.t = 0.

  def reset(self):
    self.bullet_client.resetSimulation()

  def save_state(self):
    stateID = self.bullet_client.saveState()
    jpos, _, _ = self.get_joint_states()
    stateVec = jpos[0:pandaNumDofs]
    state = {"stateID": stateID, "stateVec": np.array(stateVec)}
    return state

  def restore_state(self, state):
    stateID = state["stateID"]
    self.bullet_client.restoreState(stateID)

  def step(self):
    self.bullet_client.stepSimulation()
    self.t += SimTimeStep

  def set_joint_values(self, joint_values):
    for j in range(pandaNumDofs):
      self.bullet_client.resetJointState(self.panda, j, joint_values[j])

  def execute(self, v):
    """
    Control the robot by Jacobian-based projection.
    args: v: The robot’s desired Cartesian velocities in 3D
             (including both translational and rotational)
             Type: numpy.ndarray of shape (6,)
    """
    vx = v.reshape(-1, 1) # target velocity of the end-effector
    self.step()
    J = self.get_jacobian_matrix()
    vq = np.ravel(np.linalg.pinv(J) @ vx)
    self.bullet_client.setJointMotorControlArray(self.panda,
                                                 range(pandaNumDofs),
                                                 self.bullet_client.VELOCITY_CONTROL,
                                                 targetVelocities=vq)
    self.step()

  def get_joint_states(self):
    jstates = self.bullet_client.getJointStates(self.panda, 
                                                range(self.pandaNumJoints))
    jpos = [state[0] for state in jstates]
    jvel = [state[1] for state in jstates]
    jtorq = [state[3] for state in jstates]
    return jpos, jvel, jtorq

  def get_motor_joint_states(self):
    jstates = self.bullet_client.getJointStates(self.panda, 
                                                range(self.bullet_client.getNumJoints(self.panda)))
    jinfos = [self.bullet_client.getJointInfo(self.panda, i) for i in range(self.pandaNumJoints)]
    jstates = [j for j, i in zip(jstates, jinfos) if i[3] > -1]
    jpos = [state[0] for state in jstates]
    jvel = [state[1] for state in jstates]
    jtorq = [state[3] for state in jstates]
    return jpos, jvel, jtorq

  def get_jacobian_matrix(self):
    """
    Get the jacobian matrix of the robot at its current configuration.
    returns: J: The jacobian matrix.
                Type: numpy.ndarray of shape (6, 7)
    """
    mjpos, _, _ = self.get_motor_joint_states()
    Jt, Jr = self.bullet_client.calculateJacobian(self.panda, pandaEndEffectorIndex, 
                                                  localPosition=[0.0, 0.0, 0.0], 
                                                  objPositions=mjpos,
                                                  objVelocities=[0.0]*len(mjpos),
                                                  objAccelerations=[0.0]*len(mjpos))
    Jt, Jr = np.array(Jt)[:, 0:pandaNumDofs], np.array(Jr)[:, 0:pandaNumDofs]
    J = np.vstack((Jt, Jr))
    return J

  def is_collision(self):
    """
    Check if there is collision between the robot and obstacles.
    returns: True of False.
    """
    if len(self.bullet_client.getContactPoints(self.panda, self.env)) > 0 \
    or len(self.bullet_client.getContactPoints(self.panda, self.plane)) > 0:
      return True
    return False

  def is_touch(self):
    """
    Check if there is contact between the stick's spherical end and any cylinder obstacle.
    returns: True of False.
    """
    pts = self.bullet_client.getContactPoints(self.panda, self.env, linkIndexA=13)
    if len(pts) > 0:
      return True
    return False
