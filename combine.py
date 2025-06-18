import trimesh

# 加载 scene
scene = trimesh.load("demo_data/tube75/mesh/tube75.obj")

# 推荐的新方式合并所有子 mesh
mesh_list = list(scene.geometry.values())
mesh = trimesh.util.concatenate(mesh_list)

# 保存为新的 .obj 文件（含颜色）
mesh.export("tube75_combined.obj")