from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import trimesh

import Utils
import estimater
from estimater import *
from datareader import *


def _material_image_or_none(material):
  image = getattr(material, "image", None)
  if image is not None:
    return image

  base_color_texture = getattr(material, "baseColorTexture", None)
  if base_color_texture is None:
    base_color_texture = getattr(material, "base_color_texture", None)
  if base_color_texture is not None:
    return getattr(base_color_texture, "image", None)

  return None


def make_mesh_tensors_compatible(mesh, device='cuda', max_tex_size=None):
  mesh_tensors = {}
  if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
    image = _material_image_or_none(mesh.visual.material)
    if image is not None:
      img = np.array(image.convert('RGB'))[..., :3]
      if max_tex_size is not None:
        max_size = max(img.shape[0], img.shape[1])
        if max_size > max_tex_size:
          scale = 1 / max_size * max_tex_size
          img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
      mesh_tensors['tex'] = torch.as_tensor(img, device=device, dtype=torch.float)[None] / 255.0
      mesh_tensors['uv_idx'] = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
      uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
      uv[:, 1] = 1 - uv[:, 1]
      mesh_tensors['uv'] = uv
    else:
      base_color = getattr(mesh.visual.material, "baseColorFactor", None)
      if base_color is None:
        base_color = getattr(mesh.visual.material, "main_color", None)
      if base_color is None:
        base_color = np.array([128, 128, 128, 255], dtype=np.uint8)
      base_color = np.asarray(base_color, dtype=np.uint8).reshape(-1)
      if len(base_color) == 3:
        base_color = np.concatenate([base_color, np.array([255], dtype=np.uint8)])
      mesh.visual.vertex_colors = np.tile(base_color.reshape(1, 4), (len(mesh.vertices), 1))
      mesh_tensors['vertex_color'] = torch.as_tensor(
        mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float
      ) / 255.0
  else:
    if mesh.visual.vertex_colors is None:
      logging.info("WARN: mesh doesn't have vertex_colors, assigning a pure color")
      mesh.visual.vertex_colors = np.tile(np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1))
    mesh_tensors['vertex_color'] = torch.as_tensor(
      mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float
    ) / 255.0

  mesh_tensors.update({
    'pos': torch.tensor(mesh.vertices, device=device, dtype=torch.float),
    'faces': torch.tensor(mesh.faces, device=device, dtype=torch.int),
    'vnormals': torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
  })
  return mesh_tensors


def load_mesh_compatible(mesh_file: str, mesh_scale: float):
  mesh = trimesh.load(mesh_file, force='scene')

  if isinstance(mesh, trimesh.Scene):
    meshes = []
    for geom in mesh.geometry.values():
      if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
        meshes.append(geom)
    if len(meshes) == 0:
      raise RuntimeError(f'No valid mesh found in {mesh_file}')
    mesh = trimesh.util.concatenate(meshes)

  if not isinstance(mesh, trimesh.Trimesh):
    raise RuntimeError(f'Failed to load a Trimesh from {mesh_file}, got {type(mesh)}')

  mesh.remove_unreferenced_vertices()
  mesh.remove_degenerate_faces()
  mesh.remove_duplicate_faces()
  mesh.process(validate=True)

  if mesh_scale <= 0:
    raise ValueError(f'--mesh_scale must be positive, got {mesh_scale}')
  mesh.apply_scale(mesh_scale)
  _ = mesh.vertex_normals
  return mesh


def save_maniskill_export(export_dir: Path, frame_id: str, mesh_file: str, mesh_scale: float, T_cam_obj: np.ndarray, T_base_cam: np.ndarray | None):
  export_dir.mkdir(parents=True, exist_ok=True)
  payload = {
    "frame_id": frame_id,
    "mesh_file": mesh_file,
    "mesh_scale": mesh_scale,
    "T_cam_obj": T_cam_obj.tolist(),
  }
  if T_base_cam is not None:
    T_base_obj = np.linalg.inv(T_base_cam) @ T_cam_obj
    payload["T_base_cam"] = T_base_cam.tolist()
    payload["T_base_obj"] = T_base_obj.tolist()

  with (export_dir / f"{frame_id}.json").open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

  with (export_dir / "latest.json").open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)


Utils.make_mesh_tensors = make_mesh_tensors_compatible
estimater.make_mesh_tensors = make_mesh_tensors_compatible


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--mesh_scale', type=float, default=1.0)
  parser.add_argument('--camera_extrinsic_opencv', type=str, default=None, help='Optional T_base_cam 4x4 .npy/.txt path')
  parser.add_argument('--export_dir', type=str, default=None, help='Optional export dir for ManiSkill bridge json')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = load_mesh_compatible(args.mesh_file, args.mesh_scale)
  logging.info(f'Using mesh_scale={args.mesh_scale}')

  T_base_cam = None
  if args.camera_extrinsic_opencv is not None:
    path = Path(args.camera_extrinsic_opencv)
    if path.suffix == ".npy":
      T_base_cam = np.load(path)
    else:
      T_base_cam = np.loadtxt(path)
    T_base_cam = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  if args.export_dir is not None:
    export_dir = Path(args.export_dir)
  else:
    scene_name = Path(args.test_scene_dir).resolve().name
    export_dir = Path(debug_dir) / "maniskill_export" / scene_name

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
    save_maniskill_export(export_dir, reader.id_strs[i], args.mesh_file, args.mesh_scale, pose.reshape(4,4), T_base_cam)

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)

    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
