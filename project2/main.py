import os
import numpy as np
import argparse
import trimesh
import alg
import utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, choices=["bunny", "cow", "duck"], default="bunny")
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    # load the mesh file and visualize
    mesh_path = "./meshes/%s.stl" % args.mesh
    mesh = trimesh.load(mesh_path)
    print("The mesh file was loaded by the path: %s" % os.path.abspath(mesh_path))
    print("Information of the mesh:")
    print("  Number of Faces: %d" % len(mesh.faces))
    print("  Number of Vertices: %d" % len(mesh.vertices))
    print("  The Center of Mass:", mesh.center_mass)
    print("\n")
    #_ = utils.plot_mesh(mesh)

    # Task 1: Grasp Quality Evaluation 
    if args.task == 1:
        grasp = [81, 267, 480]
        print("The grasp:", grasp)
        con_pts = utils.get_centroid_of_triangles(mesh, grasp)
        print("The contact points of the given grasp:")
        print(con_pts)
        utils.plot_grasp(mesh, grasp)
        Q = alg.eval_Q(mesh, grasp)
        print("The quality of the given grasp: %f \n" % Q)
    
    # Task 2: Sample a Stable Grasp
    elif args.task == 2:
        grasp, Q = alg.sample_stable_grasp(mesh)
        print("The grasp:", grasp)
        con_pts = utils.get_centroid_of_triangles(mesh, grasp)
        print("The contact points of the stable grasp:")
        print(con_pts)
        print("The quality of the given grasp: %f \n" % Q)
        utils.plot_grasp(mesh, grasp)

    # Task 3: Optimize the Given Grasp
    elif args.task == 3:
        grasp = [80, 165, 444]
        traj = alg.optimize_grasp(mesh, grasp)
        print("The quality of the given initial grasp: %f" % alg.eval_Q(mesh, traj[0]))
        print("The quality of the optimized grasp: %f" % alg.eval_Q(mesh, traj[-1]))
        utils.plot_traj(mesh, traj)

    # Task 4: Sample and Optimize a Grasp under Reachability Constraint
    elif args.task == 4:
        traj = alg.optimize_reachable_grasp(mesh, r=1.0)
        print("The quality of the given initial grasp: %f" % alg.eval_Q(mesh, traj[0]))
        print("The quality of the optimized grasp: %f" % alg.eval_Q(mesh, traj[-1]))
        utils.plot_traj(mesh, traj)
