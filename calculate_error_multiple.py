import os
import glob
import json
import copy
import numpy as np
import transforms3d.quaternions as tq
import trimesh
import open3d as o3d


def make_axis_lineset(transform: np.ndarray, size: float = 0.1) -> o3d.geometry.LineSet:
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


def load_gt_poses(json_path: str):
    data = json.load(open(json_path, 'r'))
    poses = []
    for p in data['poses']:
        pos = np.array([p['position']['x'], p['position']['y'], p['position']['z']], dtype=float)
        quat = np.array([p['orientation']['w'], p['orientation']['x'],
                         p['orientation']['y'], p['orientation']['z']], dtype=float)
        poses.append((pos, quat))
    return poses


def compute_pose_error(t_pred, q_pred, t_gt, q_gt):
    trans_err = np.linalg.norm(t_pred - t_gt) * 1000.0  # m → mm
    dot = abs(np.dot(q_pred / np.linalg.norm(q_pred), q_gt / np.linalg.norm(q_gt)))
    dot = np.clip(dot, -1.0, 1.0)
    rot_err = 2 * np.degrees(np.arccos(dot))
    return trans_err, rot_err


def main():
    est_files = sorted(glob.glob('debug/ob_in_cam/*_obj*.txt'))
    if not est_files:
        raise FileNotFoundError("No estimated pose files found.")

    first_est = os.path.basename(est_files[0])
    frame_prefix = first_est.split('_obj')[0]  # e.g., 'rgb_image_1750619764_562324797'
    frame_id = frame_prefix.replace('rgb_image_', '')
    gt_path = os.path.join('demo_data/tube75/tube_poses', f'tube_poses_{frame_id}.json')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt_poses = load_gt_poses(gt_path)
    mesh_file = 'demo_data/tube75/mesh/tube75_combined.obj'
    mesh_tr = trimesh.load(mesh_file)
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh_tr.vertices),
        o3d.utility.Vector3iVector(mesh_tr.faces)
    )
    mesh_o3d.compute_vertex_normals()

    used_gt_indices = set()
    geometries = [make_axis_lineset(np.eye(4), size=0.2)]
    print(f"\n=== Frame {frame_id} ===\n")

    for est_file in est_files:
        T_pred = np.loadtxt(est_file).reshape(4, 4)
        R_pred = T_pred[:3, :3]
        t_pred = T_pred[:3, 3]
        q_pred = tq.mat2quat(R_pred)

        best_gt_idx = -1
        best_trans_err = float('inf')
        best_rot_err = float('inf')

        for i, (t_gt, q_gt) in enumerate(gt_poses):
            if i in used_gt_indices:
                continue
            trans_err, rot_err = compute_pose_error(t_pred, q_pred, t_gt, q_gt)
            if trans_err < best_trans_err:
                best_trans_err = trans_err
                best_rot_err = rot_err
                best_gt_idx = i

        if best_gt_idx == -1:
            print(f"{est_file} → No GT matched.")
            continue

        used_gt_indices.add(best_gt_idx)
        t_gt, q_gt = gt_poses[best_gt_idx]

        print(f"{os.path.basename(est_file)}:")
        print(f"  Translation error: {best_trans_err:.1f} mm")
        print(f"  Rotation error:    {best_rot_err:.2f}°")

        T_gt = np.eye(4)
        T_gt[:3, :3] = tq.quat2mat(q_gt)
        T_gt[:3, 3] = t_gt
        mesh_gt = copy.deepcopy(mesh_o3d)
        mesh_gt.paint_uniform_color([0, 1, 0])
        mesh_gt.transform(T_gt)
        axes_gt = make_axis_lineset(T_gt, size=0.1)

        mesh_pred = copy.deepcopy(mesh_o3d)
        mesh_pred.paint_uniform_color([1, 0, 0])
        mesh_pred.transform(T_pred)
        axes_pred = make_axis_lineset(T_pred, size=0.1)

        geometries.extend([mesh_gt, axes_gt, mesh_pred, axes_pred])
        # break
    o3d.visualization.draw_geometries(geometries, window_name=f"Frame {frame_id} GT vs Predictions")


if __name__ == "__main__":
    main()
