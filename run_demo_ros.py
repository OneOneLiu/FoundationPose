#!/usr/bin/env python3
import os
import threading
import logging

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

import trimesh
import numpy as np
import cv2
import imageio

import datareader as dr
from estimater import (
    ScorePredictor,
    PoseRefinePredictor,
    FoundationPose,
    set_logging_format,
    set_seed,
    depth2xyzmap,
    toOpen3dCloud,
    draw_posed_3d_box,
    draw_xyz_axis,
)


def cleanup_debug_dir(debug_dir):
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')


class FoundationPoseNode(Node):
    def __init__(self):
        super().__init__('foundation_pose_node')
        self.started = False

        # Subscriber: only start when receives '1'
        self.create_subscription(
            Int32, '/pose_start', self.start_cb, 10
        )
        self.get_logger().info('FoundationPoseNode ready, waiting for /pose_start=1')

        # Declare parameters (with defaults matching demo)
        code_dir = os.path.dirname(os.path.realpath(__file__))
        self.declare_parameter('mesh_file', f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
        self.declare_parameter('test_scene_dir', f'{code_dir}/demo_data/mustard0')
        self.declare_parameter('est_refine_iter', 5)
        self.declare_parameter('track_refine_iter', 2)
        self.declare_parameter('debug', 1)
        self.declare_parameter('debug_dir', f'{code_dir}/debug')

    def start_cb(self, msg: Int32):
        if msg.data == 1 and not self.started:
            self.started = True
            self.get_logger().info('Received start signal, beginning pose estimation demo...')
            # Run in a background thread to avoid blocking the executor
            threading.Thread(target=self.run_demo, daemon=True).start()
        elif msg.data == 1:
            self.get_logger().info('Pose estimation already running.')

    def run_demo(self):
        # Read parameters
        mesh_file = self.get_parameter('mesh_file').value
        test_scene_dir = self.get_parameter('test_scene_dir').value
        est_refine_iter = self.get_parameter('est_refine_iter').value
        track_refine_iter = self.get_parameter('track_refine_iter').value
        debug = self.get_parameter('debug').value
        debug_dir = self.get_parameter('debug_dir').value

        # Logging & RNG
        set_logging_format()
        set_seed(0)

        # Load mesh
        # t0 = time.time() if hasattr(time, 'time') else None
        mesh = trimesh.load(mesh_file)
        self.get_logger().info(f'Mesh loaded from {mesh_file}')

        # Prepare debug dirs
        cleanup_debug_dir(debug_dir)

        # Compute oriented bounding box
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        # Initialize networks & renderer
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=debug,
            glctx=glctx,
        )
        self.get_logger().info('Estimator initialized')

        # Reader for demo data
        reader = dr.YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

        # Main loop over frames
        for i in range(len(reader.color_files)):
            self.get_logger().info(f'Processing frame {i}/{len(reader.color_files)-1}')
            color = reader.get_color(i)
            depth = reader.get_depth(i)

            if i == 0:
                mask = reader.get_mask(0).astype(bool)
                pose = est.register(
                    K=reader.K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=est_refine_iter,
                )
            else:
                pose = est.track_one(
                    rgb=color,
                    depth=depth,
                    K=reader.K,
                    iteration=track_refine_iter,
                )

            # Save pose
            os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
            np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

            # Visualization
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(
                reader.K,
                img=color,
                ob_in_cam=center_pose,
                bbox=bbox,
            )
            vis = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=reader.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            cv2.imshow('Pose Demo', vis[..., ::-1])
            cv2.waitKey(1)

            if debug >= 2:
                os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

        self.get_logger().info('Pose estimation demo completed.')


def main(args=None):
    rclpy.init(args=args)
    node = FoundationPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
