"""Test: does the humanoid RL policy respond to lateral velocity (vy)?"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from simple_env import EnvConfig
import onnxruntime as ort
import mujoco

SCENE_PATH = os.path.join(ROOT, "model/bxi_elf3/bxi_elf3_scene_factory.xml")
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
data.qpos[0] = 1.0; data.qpos[1] = 2.0; data.qpos[2] = 1.1
mujoco.mj_forward(model, data)

POLICY_PATH = os.path.join(ROOT, "model/bxi_elf3/model_normal.onnx")
session = ort.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])
kp = np.array(EnvConfig.kp, dtype=np.float32)
kd = np.array(EnvConfig.kd, dtype=np.float32)
action_scale = np.array(EnvConfig.action_scale, dtype=np.float32)
num_actions = EnvConfig.num_actions
num_obs = EnvConfig.num_obs
control_decimation = EnvConfig.control_decimation

default_angles = data.qpos[7:7+num_actions].copy()
target_dof_pos = default_angles.copy()
last_action = np.zeros(num_actions, dtype=np.float32)

def quat_rotate_inverse(q, v):
    q_w, q_vec = q[-1], q[:3]
    a = v * (2.0*q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def robot_step(cmd):
    global target_dof_pos, last_action
    tau = (target_dof_pos - data.qpos[7:7+num_actions]) * kp \
          + (np.zeros_like(kp) - data.qvel[6:6+num_actions]) * kd
    data.ctrl[:num_actions] = tau
    mujoco.mj_step(model, data)
    counter = getattr(robot_step, "counter", 0)
    robot_step.counter = counter + 1
    if counter % control_decimation == 0:
        qj = data.qpos[7:7+num_actions] - default_angles
        dqj = data.qvel[6:6+num_actions]
        omega = data.qvel[3:6].astype(np.float64)
        grav = quat_rotate_inverse(
            data.sensor("Body_Quat").data[[1,2,3,0]].astype(np.float64),
            np.array([0, 0, -1]))
        obs = np.concatenate([omega, grav, qj, dqj, last_action,
                              np.array(cmd)]).astype(np.float32)
        out = session.run(None, {"obs": obs.reshape(1, num_obs)})
        last_action = out[-1].reshape(-1)
        target_dof_pos = last_action * action_scale + default_angles

# Run A: vx=0.7, vy=0, omega=0  (baseline)
start = (data.qpos[0].copy(), data.qpos[1].copy())
for _ in range(400):  # 2s at dt=0.005
    robot_step([0.7, 0.0, 0.0])
dx_a = data.qpos[0] - start[0]
dy_a = data.qpos[1] - start[1]
yaw_a = data.qpos[3:7].copy()

# Run B: vx=0.7, vy=0.5, omega=0  (lateral command)
mujoco.mj_resetDataKeyframe(model, data, 0)
data.qpos[0] = 1.0; data.qpos[1] = 2.0; data.qpos[2] = 1.1
mujoco.mj_forward(model, data)
start = (data.qpos[0].copy(), data.qpos[1].copy())
robot_step.counter = 0
for _ in range(400):
    robot_step([0.7, 0.5, 0.0])
dx_b = data.qpos[0] - start[0]
dy_b = data.qpos[1] - start[1]
yaw_b = data.qpos[3:7].copy()

print("=== vy response test (2s) ===")
print(f"Baseline vx=0.7:  Δpos=({dx_a:.3f},{dy_a:.3f})  speed={np.hypot(dx_a,dy_a)/2:.3f} m/s")
print(f"vy=0.5 added:     Δpos=({dx_b:.3f},{dy_b:.3f})  speed={np.hypot(dx_b,dy_b)/2:.3f} m/s")
print(f"Lateral diff:     Δlat={abs(dy_b-dy_a):.3f} m  Δfwd={abs(dx_b-dx_a):.3f} m")
