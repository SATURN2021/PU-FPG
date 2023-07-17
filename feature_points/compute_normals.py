import open3d as o3d
import numpy as np


def compute_normals(x, k):
    if len(x.shape) < 3:
        x = np.expand_dims(x, axis=0)
    for i in range(x.shape[0]):
        pc = x[i]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k))
        normal = np.asarray(pcd.normals)
        normal = np.expand_dims(normal, axis=0)     # normal: (1, 256, 3)
        if i == 0:
            normals = normal
        else:
            normals = np.concatenate([normals, normal])
    return normals
