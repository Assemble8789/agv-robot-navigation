"""
elf3 人形机器人: 策略行走 + 路径跟随
====================================

复用 simple_env.py 的 EnvConfig (kp/kd/action_scale/num_obs=96) 和
run_factory_full_agv_yield.py 里机器人步进/位姿读取的逻辑, 封装成独立类:

  RobotWalker: 持有 model/data (MujocoNode 的合并场景), 加载 onnx 策略,
               step(vx,vy,omega) 走一个物理子步 (PD + 每 control_decimation
               步做一次策略推理), get_pose() 读 (x,y,yaw)。
  RobotNavigator: 沿 waypoints 的 P2P 跟随器, update(x,y,yaw) 输出
               [vx, vy=0, omega]。

物理时间: 模型 timestep 设为 EnvConfig.simulation_dt (0.005s);
策略控制频率 0.005*4 = 0.02s = 50Hz (与工厂一致)。
"""

import os
import sys
import math

import numpy as np
import mujoco
import onnxruntime as ort

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BRIDGE_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))
sys.path.insert(0, REPO_ROOT)
from simple_env import EnvConfig          # noqa: E402

POLICY = os.path.join(REPO_ROOT, "model", "bxi_elf3", "model_normal.onnx")
FACTORY_SCENE = os.path.join(REPO_ROOT, "model", "bxi_elf3",
                             "bxi_elf3_scene_factory.xml")
ROBOT_RADIUS = 0.3        # 机器人近似的圆形包络 (感知用, 避障阈值参考)

from navigation.navigate import Navigator as _PtPNav   # noqa: E402


def _factory_default_angles():
    """策略的参考姿态 = 工厂场景 keyframe 的 29 个关节角 (不是 elf3 自带 keyframe)。
    从 bxi_elf3_scene_factory.xml 的 qpos 里取后 29 个数。"""
    import re
    with open(FACTORY_SCENE, encoding="utf-8") as f:
        xml = f.read()
    m = re.search(r'<key[^>]*qpos="([^"]+)"', xml)
    if not m:
        raise RuntimeError("factory scene keyframe not found")
    return np.array([float(v) for v in m.group(1).split()][7:],
                    dtype=np.float32)


class RobotWalker:
    """策略行走器: 一个 step = 一个物理子步 (0.005s)。"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.cfg = EnvConfig()
        self.session = ort.InferenceSession(
            str(POLICY), providers=["CPUExecutionProvider"])
        self.kp = np.asarray(self.cfg.kp, dtype=np.float32)
        self.kd = np.asarray(self.cfg.kd, dtype=np.float32)
        self.action_scale = np.asarray(self.cfg.action_scale, dtype=np.float32)
        self.num_actions = self.cfg.num_actions        # 29
        self.num_obs = self.cfg.num_obs                # 96
        self.control_decimation = self.cfg.control_decimation  # 4
        # 注意: 不覆盖 model.opt.timestep。 工厂 (run_factory_full_agv_yield.py)
        # 用的是模型默认 dt=0.002 + 每 4 步一次策略推理, 覆盖成 0.005 会摔。

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.counter = 0
        # 中性关节角 = 工厂场景 keyframe 站姿 (策略参考姿态; mj_resetData 默认
        # 全零、elf3 自带 keyframe 姿态都不对, 直接抓会瘫倒)
        self.default_angles = _factory_default_angles()
        self.target_dof_pos = self.default_angles.copy()

    # ── 初始化 / 状态保存 ──────────────────────────────────────────
    def reset(self, x, y, z=1.1, yaw=0.0):
        """把机器人放到 (x,y), 站立高度 z, 朝向 yaw, 中立关节角。"""
        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = z
        qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        self.data.qpos[3:7] = (qw, 0.0, 0.0, qz)
        self.data.qpos[7:7 + self.num_actions] = self.default_angles
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.target_dof_pos = self.default_angles.copy()
        self.cmd.fill(0.0)
        self.counter = 0

    def save_state(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "default_angles": self.default_angles.copy(),
            "target_dof_pos": self.target_dof_pos.copy(),
            "action": self.action.copy(),
            "obs": self.obs.copy(),
            "cmd": self.cmd.copy(),
            "counter": self.counter,
        }

    def restore_state(self, st):
        """模型重建后恢复到重建前的物理/控制状态。"""
        self.data.qpos[:] = st["qpos"]
        self.data.qvel[:] = st["qvel"]
        self.data.ctrl[:] = st["ctrl"]
        self.default_angles = st["default_angles"]
        self.target_dof_pos = st["target_dof_pos"]
        self.action = st["action"]
        self.obs = st["obs"]
        self.cmd = st["cmd"]
        self.counter = st["counter"]

    # ── 步进 ───────────────────────────────────────────────────────
    def step(self, vx, vy, omega):
        """一个物理子步 (dt=模型默认)。 每 control_decimation 步做一次策略推理
        (与 run_factory_full_agv_yield.py 的 robot_step 完全一致: 第 0/4/8... 步)。"""
        self.cmd[:] = (vx, vy, omega)
        dqj = self.data.qvel[6:6 + self.num_actions]
        tau = ((self.target_dof_pos - self.data.qpos[7:7 + self.num_actions])
               * self.kp - dqj * self.kd)
        self.data.ctrl[:self.num_actions] = tau
        mujoco.mj_step(self.model, self.data)
        if self.counter % self.control_decimation == 0:
            self._calc_obs()
            self._policy_inference()
        self.counter += 1

    def _calc_obs(self):
        qj = self.data.qpos[7:7 + self.num_actions]
        dqj = self.data.qvel[6:6 + self.num_actions]
        omega = self.data.qvel[3:6].astype(np.double)
        qj = qj - self.default_angles
        # Body_Quat 传感器输出 [w,x,y,z], 重排成 [x,y,z,w] (与工厂一致)
        quat = self.data.sensor("Body_Quat").data[[1, 2, 3, 0]].astype(np.double)
        qx, qy, qz, qw = quat
        # quat_rotate_inverse(quat, [0,0,-1]) —— 投影重力
        v = np.array([0.0, 0.0, -1.0])
        gx = v * (2.0 * qw * qw - 1.0)
        gy = np.cross([qx, qy, qz], v) * qw * 2.0
        gz = np.array([qx, qy, qz]) * np.dot([qx, qy, qz], v) * 2.0
        gravity_orientation = gx - gy + gz
        self.obs = np.concatenate(
            [omega, gravity_orientation, qj, dqj, self.action, self.cmd]
        ).astype(np.float32)

    def _policy_inference(self):
        out = self.session.run(None, {"obs": self.obs.reshape(1, -1)})
        self.action = out[-1].reshape(-1)
        self.target_dof_pos = self.action * self.action_scale + self.default_angles

    # ── 读取 ───────────────────────────────────────────────────────
    def get_pose(self):
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        return x, y, yaw

    def fell(self, thresh=0.6):
        """摔倒检测: 躯干高度低于阈值。"""
        return self.data.qpos[2] < thresh


class RobotNavigator:
    """沿 waypoints 逐点走 —— 直接复用 navigation.navigate.Navigator
    (与 run_factory_full_agv_yield.py 的 SpatialNav 相同写法)。

    update(x,y,yaw) -> [vx, vy, omega]; 到终点后 arrived=True。"""

    def __init__(self, waypoints, fwd_speed=0.7, turn_speed=1.0, arrival=0.4):
        self.wps = [(float(p[0]), float(p[1])) for p in waypoints]
        self.idx = 0
        self.arrived = not self.wps
        self.fwd_speed = fwd_speed
        self.turn_speed = turn_speed
        self.arrival = arrival
        self.offset = np.zeros(2, dtype=np.float32)   # 世界系目标点平移 (绕行 AGV)
        self._nav = None
        self._make_nav()

    def _make_nav(self):
        if self.idx < len(self.wps):
            wx, wy = self.wps[self.idx]
            self._nav = _PtPNav(wx + self.offset[0], wy + self.offset[1],
                                fwd_speed=self.fwd_speed,
                                turn_speed=self.turn_speed)
        else:
            self._nav = None

    def update(self, x, y, yaw):
        if self.idx >= len(self.wps):
            self.arrived = True
            return np.zeros(3, dtype=np.float32)
        wx, wy = self.wps[self.idx]
        if math.hypot(wx - x, wy - y) < self.arrival:
            if self.idx < len(self.wps) - 1:
                self.idx += 1
                self.offset[:] = 0.0
                self._make_nav()
            else:
                self.arrived = True
                return np.zeros(3, dtype=np.float32)
        # 把实时偏移作用到导航器目标点上 (绕行 AGV)
        if self._nav is not None:
            self._nav.target[0] = wx + self.offset[0]
            self._nav.target[1] = wy + self.offset[1]
        return self._nav.update(x, y, yaw)
