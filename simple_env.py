from dataclasses import dataclass
import time

import mujoco.viewer
import mujoco
import numpy as np
import onnxruntime as ort
from loop_rate_limiters import RateLimiter


@dataclass(frozen=True)
class EnvConfig:
    model_xml: str = "model/bxi_elf3/bxi_elf3_scene.xml"
    #model_xml: str = "model/bxi_elf3/bxi_elf3_scene_four_boxes.xml"
    #policy: str = "model/bxi_elf3/velocity_test.onnx"
    policy: str = "model/bxi_elf3/model_normal.onnx"
    # Simulation time step
    simulation_dt: float = 0.005
    # Controller update frequency (meets the requirement of simulation_dt * control_decimation=0.02; 50Hz)
    control_decimation: int = 4
    num_actions: int = 29
    num_obs: int = 96
    actor_obs_history_length: int = 1
    kp: tuple[float,...] = (
            108.448,162.672,176.421,
            176.421,176.421,154.224,176.421,33.493,21.771,
            176.421,176.421,154.224,176.421,33.493,21.771,
            54.224,54.224,16.747,54.224,16.747,16.747,16.747,
            54.224,54.224,16.747,54.224,16.747,16.747,16.747)
    kd: tuple[float,...] = (
            6.904,10.356,11.231,
            11.231,11.231,3.452,11.231,2.132,1.386,
            11.231,11.231,3.452,11.231,2.132,1.386,
            3.452,3.452,1.066,3.452, 1.066,1.066,1.066,
            3.452,3.452,1.066,3.452, 1.066,1.066,1.066)
    action_scale: tuple[float,...] = (
            0.2075, 0.1537, 0.2126, 
            0.2126, 0.2126, 0.2075, 0.2126, 0.2986, 0.3750,
            0.2126, 0.2126, 0.2075, 0.2126, 0.2986, 0.3750, 
            0.2075, 0.2075, 0.3135, 0.2075, 0.3135, 0.3135, 0.3135, 
            0.2075, 0.2075, 0.3135, 0.2075, 0.3135, 0.3135, 0.3135)


class SimpleEnv:
    def __init__(self, config_: EnvConfig):

        # initialize config variables
        self.action_scale = config_.action_scale
        self.num_actions = config_.num_actions
        self.num_obs = config_.num_obs
        self.kp = np.asarray(config_.kp, dtype=np.float32)
        self.kd = np.asarray(config_.kd, dtype=np.float32)
        self.control_decimation = config_.control_decimation
        self.actor_obs_history_length = config_.actor_obs_history_length
        self.simulation_dt = config_.simulation_dt

        # define context variables
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.obs_history = np.zeros((self.num_obs * self.actor_obs_history_length,), dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)

        # Load robot model
        self.model = mujoco.MjModel.from_xml_path(config_.model_xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.simulation_dt # overwrite the step dt

        # load initial state
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # necessary variables for control
        self.default_angles = self.data.qpos[7:7+self.num_actions].copy()
        self.default_frame = self.data.qpos[0:7].copy()
        self.target_dof_pos = self.data.qpos[7:7+self.num_actions].copy()

        # load policy
        self.session = ort.InferenceSession(str(config_.policy), providers=["CPUExecutionProvider"])
        # time control for viewer sync
        self.rate = RateLimiter(frequency=1.0 / config_.simulation_dt, warn=False)
        self.counter = 0
    
    @staticmethod
    def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    @staticmethod
    def pd_control(target_q, q, kp, target_dq, dq, kd):
        """Calculates torques from position commands"""
        return (target_q - q) * kp + (target_dq - dq) * kd
    
    def reset(self):
        # reset the simulation to initial state
        self.data.qpos[7:7+self.num_actions] = self.default_angles.copy()
        self.data.qpos[0:7] = self.default_frame.copy()
        mujoco.mj_forward(self.model, self.data)

        # reset control variables
        self.cmd.fill(0)

        # reset control target
        self.target_dof_pos = self.default_angles.copy()

    def step(self):
        # apply PD control to get torques
        tau = self.pd_control(self.target_dof_pos, self.data.qpos[7:7+self.num_actions], self.kp, np.zeros_like(self.kp), self.data.qvel[6:6+self.num_actions], self.kd)
        self.data.ctrl[:self.num_actions] = tau

        # step the simulation
        mujoco.mj_step(self.model, self.data)
        self.counter += 1

    def calc_obs(self):
        # create observation (only for controlled joints)
        qj = self.data.qpos[7:7+self.num_actions]
        dqj = self.data.qvel[6:6+self.num_actions]
        omega = self.data.qvel[3:6].astype(np.double)

        qj = qj - self.default_angles
        gravity_orientation = self.quat_rotate_inverse(self.data.sensor("Body_Quat").data[[1, 2, 3, 0]].astype(np.double), np.array([0, 0, -1]))

        # base_ang_vel: 3
        # projected_gravity: 3
        # joint_pos_rel: 29
        # joint_vel_rel: 29
        # last_action: 29
        # command: 3
        # Total = 96
        self.obs = np.concatenate(
            [
                omega,  # 3
                gravity_orientation,  # 3
                qj, # 29
                dqj,  # 29
                self.action,  # 29 
                self.cmd,  # 3
            ],
            axis=0,
        ).astype(np.float32)

        # update obs history
        self.obs_history = np.roll(self.obs_history, shift=-self.num_obs)
        self.obs_history[-self.num_obs :] = self.obs.copy()
    
    def policy_inference(self):
        # policy inference
        model_input = {'obs': self.obs_history.reshape(1,self.num_obs * self.actor_obs_history_length)}
        model_outputs = self.session.run(None, model_input)
        self.action = model_outputs[-1].reshape(-1)
        # transform action to target_dof_pos
        self.target_dof_pos = self.action * self.action_scale + self.default_angles
    
    def print_speed(self):
        """Print commanded vs actual base velocity (body frame: fwd/lat)."""
        vx_w, vy_w, _ = self.data.qvel[0:3]
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        c, s = np.cos(yaw), np.sin(yaw)
        fwd = vx_w*c + vy_w*s
        lat = -vx_w*s + vy_w*c
        speed = np.linalg.norm(self.data.qvel[0:3])
        print(f"\rc={self.counter:>6}  cmd=({self.cmd[0]:+.2f},{self.cmd[1]:+.2f},"
              f"{self.cmd[2]:+.2f})  speed={speed:.3f}  fwd={fwd:+.3f}  "
              f"lat={lat:+.3f} m/s", end="", flush=True)

    def render_run(self,cmd: np.ndarray):
        with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False) as viewer:
            while viewer.is_running():
                self.step()

                if self.counter % self.control_decimation == 0:
                    self.cmd = cmd.copy()
                    self.calc_obs()
                    self.policy_inference()

                if self.counter % 100 == 0:   # ~0.5s
                    self.print_speed()

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                self.rate.sleep()
    
    def estimate_trajectory(self, cmd: np.ndarray, steps: int):
        traj = []
        for _ in range(steps):
            self.step()
            traj.append([self.counter*self.simulation_dt]+self.data.qpos.copy().tolist())
            if self.counter % self.control_decimation == 0:
                self.cmd = cmd.copy()
                self.calc_obs()
                self.policy_inference()
        # speed summary (fwd/lat decomposed, body frame)
        if steps > 0:
            dx = self.data.qpos[0] - traj[0][1]
            dy = self.data.qpos[1] - traj[0][2]
            dt_tot = steps * self.simulation_dt
            qw, qx, qy, qz = self.data.qpos[3:7]
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
            c, s = np.cos(yaw), np.sin(yaw)
            fwd = (dx*c + dy*s) / dt_tot
            lat = (-dx*s + dy*c) / dt_tot
            print(f"\n  cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})  over {dt_tot:.1f}s:  "
                  f"avg_fwd={fwd:+.3f}  avg_lat={lat:+.3f} m/s", flush=True)
        return np.array(traj)

if __name__ == "__main__":
    my_config = EnvConfig()
    my_env = SimpleEnv(my_config)
    # Example command: move 0.5m/s in x direction, no movement in y and omega yaw turning)
    cmd = np.array([0.5, 0.4, 0])
    traj = my_env.estimate_trajectory(cmd, steps=20)
    print("Estimated trajectory (time, wxyz_xyz, 29 joint angles):")
    print(traj)  # t, wxyz_xyz + 29 joint angles

    # run with viewer
    my_env.reset()
    my_env.render_run(cmd)
