import numpy as np
import jac
import time


# Parameters, PLEASE DO NOT CHANGE
useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11
pandaFingerJoint1Index = 9
pandaFingerJoint2Index = 10
pandaNumDofs = 7

pandaJointRange = np.array([[-2.8973, 2.8973],
                            [-1.7628, 1.7628],
                            [-2.8973, 2.8973],
                            [-3.0718, -0.0698],
                            [-2.8973, 2.8973],
                            [-0.0175, 3.7525],
                            [-2.8973, 2.8973]]) # the range of the robot's joint angles

pandaStartJoints = [0.3360771289431801, 0.33773759795964864, 0.4331145290389591,
                    -2.726785252069466, -0.9315032963941317, 2.962324195138054,
                    2.46586065658275] # start joint angles of the robot

SimTimeStep = 1./60. # the time step of the simulator


class PandaSim(object):
  """
  The simulation environment of the 7-DoF Franka Panda robot based on pybullet.
  """

  def __init__(self, bullet_client):
    """
    args: bullet_client: The client of pybullet simulator
                         Type: pybullet_utils.bullet_client.BulletClient
    """
    self.bullet_client = bullet_client
    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    self.plane = self.bullet_client.loadURDF("plane.urdf")
    visWsID = self.bullet_client.createVisualShape(self.bullet_client.GEOM_BOX,
                                                   halfExtents=[0.3, 0.3, 1e-3],
                                                   rgbaColor=[0.0, 1.0, 0.0, 1.0])
    self.bullet_client.createMultiBody(baseMass=0, baseVisualShapeIndex=visWsID)
    visGoalID = self.bullet_client.createVisualShape(self.bullet_client.GEOM_CYLINDER,
                                                     radius=0.1, length=1e-3,
                                                     rgbaColor=[1.0, 0.0, 0.0, 1.0])
    self.bullet_client.createMultiBody(baseMass=0, baseVisualShapeIndex=visGoalID, basePosition=[0.2, -0.2, 0])

    # setup the Panda robot arm
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", [-0.4, -0.2, 0.0], [0, 0, 0, 1], useFixedBase=True, flags=flags)
    self.pandaNumJoints = self.bullet_client.getNumJoints(self.panda)
    self.bullet_client.resetJointState(self.panda, pandaFingerJoint1Index, 0.04)
    self.bullet_client.resetJointState(self.panda, pandaFingerJoint2Index, 0.04)
    for j in range(pandaNumDofs):
      self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0,
                                        jointLowerLimit=pandaJointRange[j, 0], jointUpperLimit=pandaJointRange[j, 1])
      self.bullet_client.resetJointState(self.panda, j, pandaStartJoints[j])

    self.objects = []
    self.num_objects = 0

    self.obstacles = []
    self.num_obstacles = 0
    
    self.jac_solver = jac.JacSolver() # The jacobian solver for the robot
    self.pdef = None # task-specific ProblemDefinition

  def add_box(self, halfExtents, rgbaColor, pos):
    colBoxID = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_BOX,
                                                       halfExtents=halfExtents)
    visBoxID = self.bullet_client.createVisualShape(self.bullet_client.GEOM_BOX,
                                                    halfExtents=halfExtents,
                                                    rgbaColor=rgbaColor)
    box = self.bullet_client.createMultiBody(baseMass=0.1,
                                             baseCollisionShapeIndex=colBoxID,
                                             baseVisualShapeIndex=visBoxID,
                                             basePosition=[pos[0], pos[1], 0.02])
    self.bullet_client.changeDynamics(box, -1, lateralFriction=0.1)
    return box

  def add_object(self, halfExtents, rgbaColor, pos):
    """
    Add one movable box object in the simulation.
    """
    self.objects.append(self.add_box(halfExtents, rgbaColor, pos))
    self.num_objects += 1

  def add_obstacle(self, halfExtents, rgbaColor, pos):
    """
    Add one fixed box object in the simulation, as an obstacle.
    """
    self.obstacles.append(self.add_box(halfExtents, rgbaColor, pos))
    self.num_obstacles += 1

  def reset(self):
    self.bullet_client.resetSimulation()

  def set_pdef(self, pdef):
    """
    Set the Problem Definition.
    """
    self.pdef = pdef

  def get_pdef(self):
    """
    Get the Problem Definition.
    """
    return self.pdef

  def save_state(self):
    """
    Save and return the current state of the simulation. 
    """
    stateID = self.bullet_client.saveState()
    jpos, _, _ = self.get_joint_states()
    stateVec = jpos[0:pandaNumDofs]
    for i in range(self.num_objects):
      obj = self.objects[i]
      pos, quat = self.bullet_client.getBasePositionAndOrientation(obj)
      orn = self.bullet_client.getEulerFromQuaternion(quat)
      stateVec = stateVec + [pos[0], pos[1], orn[2]]
    state = {"stateID": stateID, "stateVec": np.array(stateVec)}
    return state

  def restore_state(self, state):
    """
    Restore the simulation to "state".
    """
    stateID = state["stateID"]
    self.bullet_client.restoreState(stateID)

  def open_gripper(self):
    """
    Send control command to open the gripper.
    """
    self.bullet_client.setJointMotorControlArray(self.panda, 
                                                 [9, 10], 
                                                 self.bullet_client.POSITION_CONTROL,
                                                 [0.04, 0.04], forces=[10, 10])

  def close_gripper(self):
    """
    Send control command to close the gripper.
    """
    self.bullet_client.setJointMotorControlArray(self.panda, 
                                                 [9, 10], 
                                                 self.bullet_client.POSITION_CONTROL,
                                                 [0.02, 0.02], forces=[1, 1])

  def grasp(self):
    """
    Simulate the robot's grasp.
    """
    q, _, _ = self.get_joint_states()
    q = np.array(q[0:7])
    self.bullet_client.setJointMotorControlArray(self.panda, range(pandaNumDofs), 
                                                 self.bullet_client.POSITION_CONTROL, targetPositions=q)
    self.close_gripper()
    for _ in range(200):
      self.step()
      time.sleep(0.01)
       
  def step(self):
    """
    Step the simulation.
    """
    for box in self.objects:
      self.bullet_client.applyExternalForce(box, -1, [0, 0, -0.98], [0, 0, 0], self.bullet_client.LINK_FRAME)
    self.bullet_client.stepSimulation()

  def execute(self, ctrl, sleep_time=0.0):
    """
    Control the robot by Jacobian-based projection.
    args:       ctrl: The robot’s Cartesian velocity in 2D and its duration
                      [x_dot, y_dot, theta_dot, duration]
                      Type: numpy.ndarray of shape (4,)
          sleep_time: sleep time for slowing down the simulation rendering.
                      (you don’t need to worry about this parameter)
    returns:    wpts: Intermediate waypoints of the Cartesian trajectory in the 2D space.
                      Type: numpy.ndarray of shape (number of waypoints, 3)
               valid: True or False, indicating whether all intermediate 
                      states are valid and high-quality.
    """
    valid = True
    wpts = np.empty(shape=(0, 3))
    d = ctrl[3] # duration
    vx = np.array([ctrl[0], ctrl[1], 0, 0, 0, ctrl[2]]) # desired twist of the end-effector
    
    n_steps = int(d / SimTimeStep)
    for i in range(n_steps):

      ########## TODO ##########
      J = np.zeros(shape=(6, 7)) # Jacobian matrix
      vq = np.zeros(shape=(7,)) # joint velocities


      ##########################

      if not self.pdef.is_state_high_quality(J):
        valid = False
        break

      self.bullet_client.setJointMotorControlArray(self.panda,
                                                   range(pandaNumDofs),
                                                   self.bullet_client.VELOCITY_CONTROL,
                                                   targetVelocities=vq)
      self.step()

      if (i + 1) % 12 == 0: # check for every 0.2 second
        if not self.pdef.is_state_valid(self.save_state()):
          valid = False
          break

      pos_ee, quat_ee = self.get_ee_pose()
      euler_ee = self.bullet_client.getEulerFromQuaternion(quat_ee)
      wpt = np.array([pos_ee[0], pos_ee[1], euler_ee[2] % (2 * np.pi)])
      wpts = np.vstack((wpts, wpt.reshape(1, -1)))
      time.sleep(sleep_time)
    return wpts, valid

  def get_joint_states(self):
    """
    Get state of all joints."
    """
    jstates = self.bullet_client.getJointStates(self.panda, 
                                                range(self.pandaNumJoints))
    jpos = [state[0] for state in jstates]
    jvel = [state[1] for state in jstates]
    jtorq = [state[3] for state in jstates]
    return jpos, jvel, jtorq

  def get_motor_joint_states(self):
    """
    Get state of all motor joints.
    """
    jstates = self.bullet_client.getJointStates(self.panda, 
                                                range(self.bullet_client.getNumJoints(self.panda)))
    jinfos = [self.bullet_client.getJointInfo(self.panda, i) for i in range(self.pandaNumJoints)]
    jstates = [j for j, i in zip(jstates, jinfos) if i[3] > -1]
    jpos = [state[0] for state in jstates]
    jvel = [state[1] for state in jstates]
    jtorq = [state[3] for state in jstates]
    return jpos, jvel, jtorq

  def get_jacobian_matrix(self, joint_values):
    """
    Get the jacobian matrix of the robot at the query joint values,
    by the student's implemented jacobian solver.
    returns: The jacobian matrix
             Type: numpy.ndarray of shape (6, 7)
    """
    return self.jac_solver.get_jacobian_matrix(joint_values)

  def get_ee_pose(self):
    """
    Get the pose of the end-effector.
    returns:  pos: The position of the end-effector.
                   Type: numpy.ndarray of shape (3,)
             quat: The orientation of the end-effector, represented by quaternion.
                   Type: numpy.ndarray of shape (4,)
    """
    ee_state = self.bullet_client.getLinkState(self.panda, linkIndex=11)
    pos, quat = ee_state[4], ee_state[5]
    return np.array(pos), np.array(quat)

  def get_jacobian_matrix_online(self):
    mjpos, _, _ = self.get_motor_joint_states()
    Jt, Jr = self.bullet_client.calculateJacobian(self.panda, pandaEndEffectorIndex, 
                                                  localPosition=[0.0, 0.0, 0.0], 
                                                  objPositions=mjpos,
                                                  objVelocities=[0.0]*len(mjpos),
                                                  objAccelerations=[0.0]*len(mjpos))
    Jt, Jr = np.array(Jt)[:, 0:pandaNumDofs], np.array(Jr)[:, 0:pandaNumDofs]
    J = np.vstack((Jt, Jr))
    return J

  def is_collision(self, state):
    """
    Check if there is collision in the simulation when it is at "state".
    args: state: The state at which the collision is checked for the simulation.
                 Type: dict, {"stateID": int, "stateVec": numpy.ndarray}
    returns: True of False.
    """
    state_curr = self.save_state()
    self.restore_state(state)
    if len(self.bullet_client.getContactPoints(self.panda, self.plane)) > 0:
      self.restore_state(state_curr)
      return True
    for obs in self.obstacles:
      if len(self.bullet_client.getContactPoints(self.panda, obs)) > 0:
        self.restore_state(state_curr)
        return True
    self.restore_state(state_curr)
    return False
