import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort



def mock_inference():
	model_path = "CodeSiemens/inverse-kinematics-playgound/model/bxi_elf3/velocity_test.onnx"
	session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

	model_input = {'obs': np.random.standard_normal((1,96)).astype(np.float32)}
	model_outputs = session.run(None, model_input)
	out = model_outputs[-1].reshape(-1)
	print("Outputs:",out.shape,out)

def mock_inference2():
	model_path = "CodeSiemens/inverse-kinematics-playgound/model/bxi_elf3/amp_test.onnx"
	session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

	model_input = {'obs': np.random.standard_normal((1,1020)).astype(np.float32)}
	model_outputs = session.run(None, model_input)
	out = model_outputs[-1].reshape(-1)
	print("Outputs:",out.shape,out)

def mock_input():
	# base_ang_vel: 3
    # projected_gravity: 3
    # joint_pos_rel: 29
    # joint_vel_rel: 29
    # last_action: 29
    # command: 3
    # Total = 96
	pass

def mock_input2():
	# ang_vel: 3
    # projected_gravity: 3
    # command: 3
    # joint_pos: 29
    # joint_vel: 29
    # previous action: 29
    # sin(gait_phase): 2
    # cos(gait_phase): 2
    # phase_ratio: 2
    # Total = 102
	# History length is 10 => 1020
	pass


if __name__ == "__main__":
	mock_inference()
