from estimater import *
from datareader import *
import argparse
import os
import logging
import numpy as np
import trimesh
import imageio
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/tube75/mesh/tube75_combined.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/tube75')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    # 只创建一个 estimator 实例
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner,
                         debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    for i in range(len(reader.color_files)):
        logging.info(f'Frame: {i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        masks = reader.get_mask_multiple(i)
        poses = []
        for idx, mask in enumerate(masks):
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            poses.append(pose)
        print(f"Frame {i} - Found {len(poses)} objects")

        # 保存多个物体的估计位姿
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        for idx, pose in enumerate(poses):
            np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}_obj{idx}.txt', pose.reshape(4, 4))

        # 可视化所有物体 bbox + xyz
        if debug >= 1 and poses:
            vis = color.copy()
            for pose in poses:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=vis, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3,
                                    transparency=0, is_input_rgb=True)
            cv2.imshow('multi-object tracking', vis[..., ::-1])
            cv2.waitKey(1)

        if debug >= 2 and poses:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
