import numpy as np
import itertools as it
import scipy.spatial
import utils
import time


########## Task 1: Primitive Wrenches ##########

def primitive_wrenches(mesh, grasp, mu=0.2, n_edges=8):
    """
    Find the primitive wrenches for each contact of a grasp.
    args:   mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
           grasp: The indices of the mesh triangles being contacted.
                  Type: list of int
              mu: The friction coefficient of the mesh surface.
                  (default: 0.2)
         n_edges: The number of edges of the friction polyhedral cone.
                  Type: int (default: 8)
    returns:   W: The primitive wrenches.
                  Type: numpy.ndarray of shape (len(grasp) * n_edges, 6)
    """
    ########## TODO ##########
    W = np.zeros(shape=(len(grasp) * n_edges, 6))
    

    ##########################
    return W


########## Task 2: Grasp Quality Evaluation ##########

def eval_Q(mesh, grasp, mu=0.2, n_edges=8, lmbd=1.0):
    """
    Evaluate the L1 quality of a grasp.
    args:   mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
           grasp: The indices of the mesh triangles being contacted.
                  Type: list of int
              mu: The friction coefficient of the mesh surface.
                  (default: 0.2)
         n_edges: The number of edges of the friction polyhedral cone.
                  Type: int (default: 8)
            lmbd: The scale of torque magnitude.
                  (default: 1.0)
    returns:   Q: The L1 quality score of the given grasp.
    """
    ########## TODO ##########
    Q = -np.inf


    ##########################
    return Q


########## Task 3: Stable Grasp Sampling ##########

def sample_stable_grasp(mesh, thresh=0.0):
    """
    Sample a stable grasp such that its L1 quality is larger than a threshold.
    args:     mesh: The object mesh model.
                    Type: trimesh.base.Trimesh
            thresh: The threshold for stable grasp.
                    (default: 0.0)
    returns: grasp: The stable grasp represented by the indices of triangles.
                    Type: list of int
                 Q: The L1 quality score of the sampled grasp, 
                    expected to be larger than thresh.
    """
    ########## TODO ##########
    grasp = None
    Q = -np.inf


    ##########################
    return grasp, Q


########## Task 4: Grasp Optimization ##########

def find_neighbors(mesh, tr_id, eta=1):
    """
    Find the eta-order neighbor faces (triangles) of tr_id on the mesh model.
    args:       mesh: The object mesh model.
                      Type: trimesh.base.Trimesh
               tr_id: The index of the query face (triangle).
                      Type: int
                 eta: The maximum order of the neighbor faces:
                      Type: int
    returns: nbr_ids: The list of the indices of the neighbor faces.
                      Type: list of int
    """
    ########## TODO ##########
    nbr_ids = []



    ##########################
    return nbr_ids

def local_optimal(mesh, grasp):
    """
    Find the optimal neighbor grasp of the given grasp.
    args:     mesh: The object mesh model.
                    Type: trimesh.base.Trimesh
             grasp: The indices of the mesh triangles being contacted.
                    Type: list of int
    returns: G_opt: The optimal neighbor grasp with the highest quality.
                    Type: list of int
             Q_max: The L1 quality score of G_opt.
    """
    ########## TODO ##########
    G_opt = None
    Q_max = -np.inf


    ##########################
    return G_opt, Q_max

def optimize_grasp(mesh, grasp):
    """
    Optimize the given grasp and return the trajectory.
    args:    mesh: The object mesh model.
                   Type: trimesh.base.Trimesh
            grasp: The indices of the mesh triangles being contacted.
                   Type: list of int
    returns: traj: The trajectory of the grasp optimization.
                   Type: list of grasp (each grasp is a list of int)
    """
    traj = []
    ########## TODO ##########
   

    ##########################
    return traj


########## Task 5: Grasp Optimization with Reachability ##########

def optimize_reachable_grasp(mesh, r=0.5):
    """
    Sample a reachable grasp and optimize it.
    args:    mesh: The object mesh model.
                   Type: trimesh.base.Trimesh
                r: The reachability measure. (default: 0.5)
    returns: traj: The trajectory of the grasp optimization.
                   Type: list of grasp (each grasp is a list of int) 
    """
    traj = []
    ########## TODO ##########
   

    ##########################
    return traj
