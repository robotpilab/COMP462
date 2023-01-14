import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import sim


def setup_bullet_client(connection_mode):
  bullet_client = bc.BulletClient(connection_mode=connection_mode)
  bullet_client.resetSimulation()
  bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  bullet_client.setAdditionalSearchPath(pd.getDataPath())
  bullet_client.setTimeStep(sim.SimTimeStep)
  bullet_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1) # determinism guaranteed, important
  return bullet_client

def setup_env(panda_sim):
  # set up the environment
  panda_sim.add_object([0.02, 0.02, 0.02], [1.0, 1.0, 0.0, 1.0], [0, 0])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [-0.05, -0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0, -0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0.05, -0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [-0.05, 0])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0.05, 0])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [-0.05, 0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0, 0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0.05, 0.05])


def execute_plan(panda_sim, plan, sleep_time=0.005):
  for node in plan:
    #panda_sim.restore_state(node.state)
    p_from, _ = panda_sim.get_ee_pose()
    ctrl = node.get_control()
    if ctrl is not None:
      _ = panda_sim.execute(ctrl, sleep_time=sleep_time)
      p_to, _ = panda_sim.get_ee_pose()
      draw_line(panda_sim, p_from, p_to, c=[1, 0, 0], w=20)

def extract_reference_waypoints(panda_sim, ctrl):
  wpts_ref = np.empty(shape=(0, 3))
  d = ctrl[3]
  n_steps = int(d / sim.SimTimeStep)
  pos, quat = panda_sim.get_ee_pose()
  euler = panda_sim.bullet_client.getEulerFromQuaternion(quat)
  yaw = euler[2]
  for i in range(n_steps):
    wpt = np.array([pos[0] + (i + 1) * sim.SimTimeStep * ctrl[0],
                    pos[1] + (i + 1) * sim.SimTimeStep * ctrl[1],
                    (yaw + (i + 1) * sim.SimTimeStep * ctrl[2]) % (2 * np.pi)])
    wpts_ref = np.vstack((wpts_ref, wpt.reshape(1, -1)))
  return wpts_ref


def draw_line(panda_sim, p_from, p_to, c, w):
    return panda_sim.bullet_client.addUserDebugLine(p_from, p_to, lineColorRGB=c, lineWidth=w)
