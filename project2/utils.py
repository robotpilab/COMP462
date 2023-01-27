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
    