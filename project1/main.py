import time
import argparse
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
import sim
from pdef import Bounds, ProblemDefinition
from goal import RelocateGoal, GraspGoal
import rrt
import utils
import opt


def setup_pdef(panda_sim):
  pdef = ProblemDefinition(panda_sim)
  dim_state = pdef.get_state_dimension()
  dim_ctrl = pdef.get_control_dimension()

  # define bounds for state and control space
  bounds_state = Bounds(dim_state)
  for j in range(sim.pandaNumDofs):
    bounds_state.set_bounds(j, sim.pandaJointRange[j, 0], sim.pandaJointRange[j, 1])
  for j in range(sim.pandaNumDofs, dim_state):
    if ((j - sim.pandaNumDofs) % 3 == 2):
      bounds_state.set_bounds(j, -np.pi, np.pi)
    else:
      bounds_state.set_bounds(j, -0.3, 0.3)
  pdef.set_state_bounds(bounds_state)

  bounds_ctrl = Bounds(dim_ctrl)
  bounds_ctrl.set_bounds(0, -0.2, 0.2)
  bounds_ctrl.set_bounds(1, -0.2, 0.2)
  bounds_ctrl.set_bounds(2, -1.0, 1.0)
  bounds_ctrl.set_bounds(3, 0.4, 0.6)
  pdef.set_control_bounds(bounds_ctrl)
  return pdef


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=int, choices=[1, 2, 3, 4])
  args = parser.parse_args()

  # set up the simulation
  pgui = utils.setup_bullet_client(p.GUI)
  panda_sim = sim.PandaSim(pgui)

  # Task 1: Move the Robot with Jacobian-based Projection
  if args.task == 1:
    pdef = setup_pdef(panda_sim)

    ctrls = [[0.02, 0, 0.2, 10],
             [0, 0.02, 0.2, 10],
             [-0.02, 0, -0.2, 10],
             [0, -0.02, -0.2, 10]]
    errs = []
    for _ in range(10):
      for ctrl in ctrls:
        wpts_ref = utils.extract_reference_waypoints(panda_sim, ctrl)
        wpts, _ = panda_sim.execute(ctrl)
        err_pos = np.mean(np.linalg.norm(wpts[:, 0:2] - wpts_ref[:, 0:2], axis=1))
        err_orn = np.mean(np.abs(wpts[:, 2] - wpts_ref[:, 2]))
        print("The average Cartesian error for executing the last control:")
        print("Position: %f meters\t Orientation: %f rads" % (err_pos, err_orn))
        errs.append([err_pos, err_orn])
    errs = np.array(errs)
    print("\nThe average Cartesian error for the entire exeution:")
    print("Position: %f meters\t Orientation: %f rads" % (errs[:, 0].mean(), errs[:, 1].mean()))


  else:
    # configure the simulation and the problem
    utils.setup_env(panda_sim)
    pdef = setup_pdef(panda_sim)

    # Task 2: Kinodynamic RRT Planning for Relocating
    if args.task == 2:
      goal = RelocateGoal()
      pdef.set_goal(goal)

      planner = rrt.KinodynamicRRT(pdef)
      time_st = time.time()
      solved, plan = planner.solve(120.0)
      print("Running time of rrt.KinodynamicRRT.solve(): %f secs" % (time.time() - time_st))

      if solved:
        print("The Plan has been Found:")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan)
        while True:
          pass

    # Task 3: Kinodynamic RRT Planning for Grasping
    elif args.task == 3:
      goal = GraspGoal()
      pdef.set_goal(goal)

      planner = rrt.KinodynamicRRT(pdef)
      time_st = time.time()
      solved, plan = planner.solve(120.0)
      print("Running time of rrt.KinodynamicRRT.solve(): %f secs" % (time.time() - time_st))

      if solved:
        print("The Plan has been Found:")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan)
        panda_sim.grasp()
        while True:
          pass

    # Task 4: Trajectory Optimization
    elif args.task == 4:
      ########## TODO ##########
      pass

    
      ##########################
