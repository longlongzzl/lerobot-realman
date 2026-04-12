# RM75 cuRobo Config

This folder contains a first-pass cuRobo robot config for RM75 that is meant
to be practical, conservative, and easy to tune.

Files:

- `rm75.yml`: main cuRobo robot config
- `spheres/rm75_rough.yml`: rough hand-written collision spheres for RM75

Current modeling choices:

- Robot body: rough sphere model, intentionally a bit conservative
- Environment obstacles: keep using world cuboids / world meshes from the task pipeline
- Attached object: `attached_object` placeholder link is enabled with
  `extra_collision_spheres`, so the runtime can later populate carried-object
  spheres

Usage:

```bash
python /home/zhangzhao/Desktop/lerobot/pick_jiaobang/rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place.py \
  --curobo-rm75-robot-cfg /home/zhangzhao/Desktop/lerobot/pick_jiaobang/curobo_rm75_config/rm75.yml
```

Recommended tuning order:

1. Verify FK / IK / MotionGen still succeed with this config.
2. Check whether self-collision is too conservative.
3. Tighten or expand spheres link by link.
4. Add runtime attached-object spheres for the pen after the body model is stable.

Notes:

- `collision_spheres` uses an absolute path on purpose so it works with the
  current local planner loader without extra path-resolution changes.
- This is a starting point, not a final production collision model.
