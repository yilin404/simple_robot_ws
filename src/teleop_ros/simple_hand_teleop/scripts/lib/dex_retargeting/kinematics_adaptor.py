from abc import abstractmethod
from typing import List

import numpy as np

from .robot_wrapper import RobotWrapper


class KinematicAdaptor:
    def __init__(self, robot: RobotWrapper, target_joint_names: List[str]):
        self.robot = robot
        self.target_joint_names = target_joint_names

        # Index mapping
        self.idx_pin2target = np.array([robot.get_joint_index(n) for n in target_joint_names])

    @abstractmethod
    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        Adapt the joint position for different kinematics constraints.
        Note that the joint order of this qpos is consistent with pinocchio
        Args:
            qpos: the pinocchio qpos

        Returns: the adapted qpos with the same shape as input

        """
        pass

    @abstractmethod
    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Adapt the jacobian for different kinematics applications.
        Note that the joint order of this Jacobian is consistent with pinocchio
        Args:
            jacobian: the original jacobian

        Returns: the adapted jacobian with the same shape as input

        """
        pass


class MimicJointKinematicAdaptor(KinematicAdaptor):
    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str], # nq_active \times str
        source_joint_names: List[str], # nq_mimic \times str
        mimic_joint_names: List[str], # nq_mimic \times str
        multipliers: List[float], # nq_mimic \times float
        offsets: List[float], # nq_mimic \times float
    ):
        super().__init__(robot, target_joint_names)

        self.multipliers = np.array(multipliers)
        self.offsets = np.array(offsets)

        # Joint name check
        union_set = set(mimic_joint_names).intersection(set(target_joint_names)) # 求两个集合的交集
        if len(union_set) > 0: # 
            raise ValueError(
                f"Mimic joint should not be one of the target joints.\n"
                f"Mimic joints: {mimic_joint_names}.\n"
                f"Target joints: {target_joint_names}\n"
                f"You need to specify the target joint names explicitly in your retargeting config"
                f" for robot with mimic joint constraints: {target_joint_names}"
            )

        # Indices in the pinocchio
        self.idx_pin2source = np.array([robot.get_joint_index(name) for name in source_joint_names])
        self.idx_pin2mimic = np.array([robot.get_joint_index(name) for name in mimic_joint_names])

        # Indices in the output results
        self.idx_target2source = np.array([self.target_joint_names.index(n) for n in source_joint_names])

        # Dimension check
        len_source, len_mimic = self.idx_target2source.shape[0], self.idx_pin2mimic.shape[0]
        len_mul, len_offset = self.multipliers.shape[0], self.offsets.shape[0]
        if not (len_mimic == len_source == len_mul == len_offset):
            raise ValueError(
                f"Mimic joints setting dimension mismatch.\n"
                f"Source joints: {len_source}, mimic joints: {len_mimic}, multiplier: {len_mul}, offset: {len_offset}"
            )
        # self.num_active_joints = len(robot.dof_joint_names) - len_mimic # 计算不包含mimic关节的活动关节个数

        # Uniqueness check
        if len(mimic_joint_names) != len(np.unique(mimic_joint_names)):
            raise ValueError(f"Redundant mimic joint names: {mimic_joint_names}")

    def forward_qpos(self, 
                     pin_qpos: np.ndarray # [nq,]
                     ) -> np.ndarray:
        mimic_qpos = pin_qpos[self.idx_pin2source] * self.multipliers + self.offsets # [nq_mimic,]
        pin_qpos[self.idx_pin2mimic] = mimic_qpos # [nq_mimic,]
        return pin_qpos

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        target_jacobian = jacobian[..., self.idx_pin2target]
        mimic_joint_jacobian = jacobian[..., self.idx_pin2mimic] * self.multipliers

        for i, index in enumerate(self.idx_target2source):
            target_jacobian[..., index] += mimic_joint_jacobian[..., i]
        return target_jacobian
