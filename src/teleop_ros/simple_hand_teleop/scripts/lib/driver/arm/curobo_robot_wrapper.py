# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

import torch

class CuroboRobotWrapper:
    def __init__(self, config_file_path: str, enable_self_collision_check: bool) -> None:
        super().__init__()

        tensor_args = TensorDeviceType()
        config_file = load_yaml(config_file_path)
        if enable_self_collision_check:
            robot_cfg = RobotConfig.from_dict(config_file["robot_cfg"])

            ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                None,
                rotation_threshold=0.05,
                position_threshold=0.005,
                num_seeds=20,
                self_collision_check=True,
                self_collision_opt=True,
                tensor_args=tensor_args,
                use_cuda_graph=True,
            )
        else:
            urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
            base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
            ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
            robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

            ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                None,
                rotation_threshold=0.05,
                position_threshold=0.005,
                num_seeds=20,
                self_collision_check=False,
                self_collision_opt=False,
                tensor_args=tensor_args,
                use_cuda_graph=True,
            )
        
        self.ik_solver = IKSolver(ik_config)
    
    def forwardKinematics(self, 
                          q: torch.Tensor # [b, num_joints]
                          ) -> Pose:
        kin_state = self.ik_solver.fk(q)

        return Pose(kin_state.ee_position, kin_state.ee_quaternion) # wxyz format
    
    def inverseKinematics(self,
                          target_translation: torch.Tensor, # [b, 3] 
                          target_quaternion: torch.Tensor, # [b, 4]
                          ) -> torch.Tensor:
        goal = Pose(target_translation, target_quaternion)
        result = self.ik_solver.solve_batch(goal) # wxyz format

        q_solution = result.solution[result.success]

        return q_solution