import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def get_centroid_of_triangles(mesh, tr_ids):
    """
    Calculate the centroids of the triangles.
    args:   mesh: mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
          tr_ids: The indices of the triangles on the mesh model.
                  Type: list of int
    returns: cen: The centroids of the triangles.
                  Type: numpy.ndarray of shape (len(tr_ids), 3)          
    """
    vtx_ids = mesh.faces[tr_ids]
    centroids = []
    for vi in vtx_ids:
        vtx = mesh.vertices[vi]
        centroids.append(vtx.mean(0))
    cen = np.array(centroids)
    return cen

def check_wrenches(mesh, grasp, W, n_edges=8):
    """
    Check whether the wrench calculation is correct or not.
    args:   mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
           grasp: The indices of the mesh triangles being contacted.
                  Type: list of int  
               W: The primitive wrenches.
                  Type: numpy.ndarray of shape (len(grasp) * n_edges, 6)
          n_edges: The number of edges of the friction polyhedral cone.
                   Type: int (default: 8)  
    """
    f_success = 0
    t_success = 0
    con_pts = get_centroid_of_triangles(mesh, grasp)
    cm = mesh.center_mass
    for j in range(len(grasp)):
        F = W[j*n_edges:(j+1)*n_edges, 0:3]
        T = W[j*n_edges:(j+1)*n_edges, 3:6]
        T_scaled = np.array([t / np.linalg.norm(t) for t in T])
        n = mesh.face_normals[grasp[j]]
        con_pt = con_pts[j]
        f_sum = np.sum(F, axis=0)
        print(np.sum(n * f_sum) / np.linalg.norm(f_sum))
        if (np.sum(n * f_sum) / np.linalg.norm(f_sum) > 0.99):
            f_success += 1
        r = con_pt - cm
        torq = np.cross(r, F)
        torq_scaled = np.array([t / np.linalg.norm(t) for t in torq])
        if (np.sum(np.linalg.norm(torq_scaled - T_scaled, axis=1)) < 1e-2):
            t_success += 1
    print("\n")
    print("The correctness of the forces: ", f_success == len(grasp))
    print("The correctness of the torques: ", t_success == len(grasp))
    print("\n")

def plot_mesh(mesh, show=True):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], 
                    mesh.vertices[:, 1],
                    mesh.vertices[:, 2],
                    triangles=mesh.faces,
                    alpha=0.3)
    if show:
        plt.show()
    return ax

def plot_grasp(mesh, grasp):
    ax = plot_mesh(mesh, show=False)
    contacts = get_centroid_of_triangles(mesh, grasp)
    # plot contact points
    ax.scatter(contacts[:, 0], contacts[:, 1], contacts[:, 2],
               c='r', s=200, alpha=1, label="contact point")
    # plot normals at contact points
    _, _, tr_ids = mesh.nearest.on_surface(contacts)
    normals = mesh.face_normals[tr_ids]
    ax.quiver(contacts[:, 0], contacts[:, 1], contacts[:, 2],
              normals[:, 0], normals[:, 1], normals[:, 2], 
              length=0.3, color="g", label="normal")
    ax.legend()
    plt.show()

def plot_traj(mesh, traj):
    ax = plot_mesh(mesh, show=False)
    C_traj = [] # trajectory of contact points
    for G in traj:
        C_traj.append(get_centroid_of_triangles(mesh, G))
    C_traj = np.array(C_traj)
    ax.scatter(C_traj[0, :, 0], C_traj[0, :, 1], C_traj[0, :, 2],
               c='r', s=200, alpha=1, label="initial grasp")
    for i in range(C_traj.shape[2]):
        ax.plot(C_traj[:, i, 0], C_traj[:, i, 1], C_traj[:, i, 2],
                c="g", lw=5)
    ax.scatter(C_traj[-1, :, 0], C_traj[-1, :, 1], C_traj[-1, :, 2],
               c='y', s=200, alpha=1, label="optimized grasp")
    ax.legend()
    plt.show()
    