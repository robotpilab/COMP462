import argparse
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
import sim
import alg
import utils


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=int, choices=[1, 2, 3])
  parser.add_argument("--sigma", type=float, default=0.05)
  parser.add_argument("--num_particles", type=int, default=1000)
  args = parser.parse_args()

  # groundtruth calibration
  loc_gt = [-0.3, -0.3, 0.9]

  # load pre-collected observations
  obvs = utils.load_npy("obvs.npy")
  
  # Task 1: Caculation Weights for the Given Particles
  if args.task == 1:
    particles = np.linspace(np.array([-1, -1, -np.pi]),
                            np.array([1, 1, np.pi]),
                            num=10)
    print("The given particles are:")
    print(particles)           
    weights = alg.cal_weights(particles, obvs[0], sigma=args.sigma)
    print("\nWeights for the given particles:")
    print(weights)
    idx = alg.most_likely_particle(particles, obvs[0])
    print("\nThe index of the most likely particle is: %d" % idx)

  # Task 2: Particle Filtering with Given Observations
  if args.task == 2:
    # set up bullet client and the environment
    bullet_client = bc.BulletClient(connection_mode=p.GUI)
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    bullet_client.setAdditionalSearchPath(pd.getDataPath())
    bullet_client.setTimeStep(sim.SimTimeStep)
    bullet_client.resetSimulation()
    bullet_client.setGravity(0, 0, 0)
    panda_sim = sim.PandaSim(bullet_client, loc=loc_gt)
    est = alg.particle_filter(panda_sim, obvs, args.num_particles, sigma=args.sigma)
    print("Estimate by Particle Filtering:", est)

  # Task 3: Particle Filtering with Online Observation Generation
  if args.task == 3:
    # set up bullet client and the environment
    bullet_client = bc.BulletClient(connection_mode=p.GUI)
    bullet_client.setAdditionalSearchPath(pd.getDataPath())
    timeStep = 1./500.
    bullet_client.setTimeStep(timeStep)
    bullet_client.resetSimulation()
    bullet_client.setGravity(0, 0, 0)
    panda_sim = sim.PandaSim(bullet_client, loc=loc_gt)
    est = alg.particle_filter_online(panda_sim, args.num_particles, sigma=args.sigma)
    print("Estimate by Particle Filtering:", est)

