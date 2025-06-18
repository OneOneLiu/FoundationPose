#!/usr/bin/env python3
import os
import glob
import json
import copy
import numpy as np
import transforms3d.quaternions as tq
import trimesh
import open3d as o3d

def make_axis_lineset(transform: np.ndarray, size: float = 0.1) -> o3d.geometry.LineSet:
    """
    Return an Open3D LineSet drawing the X/Y/Z axes of a frame.
    X → red, Y → green, Z → blue.
    """
    pts = np.array([
        [0, 0, 0],
        [size, 0, 0],
        [0, size, 0],
        [0, 0, size]
    ], dtype=float)
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    ls.transform(transform)
    return ls

def load_gt(json_path: str):
    """
    Load GT position and quaternion from JSON.
    Returns (t_gt (3,), q_gt (4,)) where q_gt is [w, x, y, z].
    """
    data = json.load(open(json_path, 'r'))
    p = data['poses'][0]['position']
    q = data['poses'][0]['orientation']
    t_gt = np.array([p['x'], p['y'], p['z']], dtype=float)
    q_gt = np.array([q['w'], q['x'], q['y'], q['z']], dtype=float)
    return t_gt, q_gt

def main():
    # —— 找到预测与GT 的文件 —— 
    est_files = sorted(glob.glob('debug/ob_in_cam/*.txt'))
    if not est_files:
        raise FileNotFoundError("No prediction files in debug/ob_in_cam/")
    est_path = est_files[0]
    frame_id = os.path.basename(est_path).replace('rgb_image_', '').replace('.txt', '')
    gt_path = os.path.join('demo_data/tube75/tube_poses', f'tube_poses_{frame_id}.json')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"No GT JSON for frame {frame_id}: expected {gt_path}")

    # —— 读取预测 4x4 矩阵 —— 
    T = np.loadtxt(est_path).reshape(4,4)
    R_pred = T[:3, :3]
    t_pred = T[:3, 3]

    # —— 旋转矩阵 → 四元数 [w,x,y,z] —— 
    q_pred = tq.mat2quat(R_pred)

    # —— 读取 GT 位姿 —— 
    t_gt, q_gt = load_gt(gt_path)

    # —— 计算误差 —— 
    trans_err = np.linalg.norm(t_pred - t_gt) * 1000.0  # m → mm
    # 旋转误差 angle = 2 * acos(|dot(q1,q2)|)
    dot = abs(np.dot(q_pred/np.linalg.norm(q_pred), q_gt/np.linalg.norm(q_gt)))
    dot = np.clip(dot, -1.0, 1.0)
    rot_err = 2 * np.degrees(np.arccos(dot))

    # —— 打印 —— 
    np.set_printoptions(precision=6, suppress=True)
    print(f"\n=== Frame {frame_id} ===\n")
    print("Predicted position (m):", t_pred)
    print("Predicted quaternion [w,x,y,z]:", q_pred)
    print("GT position        (m):", t_gt)
    print("GT quaternion      [w,x,y,z]:", q_gt)
    print(f"\nTranslation error: {trans_err:.1f} mm")
    print(f"Rotation error:    {rot_err:.2f}°\n")

    # —— 可视化 —— 
    # 载入 mesh
    mesh_file = 'demo_data/tube75/mesh/tube75_combined.obj'
    mesh_tr = trimesh.load(mesh_file)
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh_tr.vertices),
        o3d.utility.Vector3iVector(mesh_tr.faces)
    )
    mesh_o3d.compute_vertex_normals()

    # GT 模型 (绿)
    T_gt = np.eye(4)
    T_gt[:3, :3] = tq.quat2mat(q_gt)
    T_gt[:3, 3]  = t_gt
    mesh_gt = copy.deepcopy(mesh_o3d)
    mesh_gt.paint_uniform_color([0, 1, 0])
    mesh_gt.transform(T_gt)
    axes_gt = make_axis_lineset(T_gt, size=0.1)

    # 预测模型 (红)
    T_pred = np.eye(4)
    T_pred[:3, :3] = tq.quat2mat(q_pred)
    T_pred[:3, 3]  = t_pred
    mesh_pred = copy.deepcopy(mesh_o3d)
    mesh_pred.paint_uniform_color([1, 0, 0])
    mesh_pred.transform(T_pred)
    axes_pred = make_axis_lineset(T_pred, size=0.1)

    # 全局原点坐标轴 (白)
    axes_origin = make_axis_lineset(np.eye(4), size=0.2)

    o3d.visualization.draw_geometries([
        mesh_gt, mesh_pred,
        axes_gt, axes_pred,
        axes_origin
    ], window_name=f"Frame {frame_id}: GT vs Pred")

if __name__ == "__main__":
    main()
