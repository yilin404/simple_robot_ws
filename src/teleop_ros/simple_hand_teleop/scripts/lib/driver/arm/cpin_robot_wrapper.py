import numpy as np
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin

import os

from typing import Tuple, List, Optional

class CPinRobotWrapper:
    def __init__(self, 
                 urdf_filename: str,
                 locked_joints: List[str],
                 ee_link: str) -> None:
        super().__init__()

        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_filename,
                                                    [os.path.join(os.path.dirname(urdf_filename), "../..")])
        self.reduced_robot = self.robot.buildReducedRobot(list_of_joints_to_lock=locked_joints,
                                                          reference_configuration=np.array([0.0] * self.robot.model.nq),)
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.cTf = casadi.SX.sym("Tf", 4, 4)

        # Get the ee_link id and define the error function
        self.ee_link_id = self.reduced_robot.model.getFrameId(ee_link)

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                self.cdata.oMf[self.ee_link_id].translation - self.cTf[:3, 3],
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                cpin.log3(self.cTf[:3, :3].T @ self.cdata.oMf[self.ee_link_id].rotation),
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_Tf = self.opti.parameter(4, 4)
        self.param_alpha = self.opti.parameter()
        self.param_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_Tf))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_Tf))
        self.smooth_cost = self.param_alpha * casadi.sumsqr(self.var_q - self.param_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)
    
    def forwardKinematics(self, 
                          q: np.ndarray # [num_joints,]
                          ) -> Tuple[np.ndarray, np.ndarray]:
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        translation = self.reduced_robot.data.oMf[self.ee_link_id].translation
        rotation = self.reduced_robot.data.oMf[self.ee_link_id].rotation

        return translation, pin.Quaternion(rotation).coeffs() # xyzw format

    def inverseKinematics(self,
                          target_translation: np.ndarray, # [3] 
                          target_quaternion: np.ndarray, # [4]
                          initial_guess: Optional[np.ndarray] = None, # [num_joints,]
                          ) -> np.ndarray:
        if initial_guess is not None:
            self.opti.set_initial(self.var_q, initial_guess)
            self.opti.set_value(self.param_q_last, initial_guess)
            self.opti.set_value(self.param_alpha, 0.1)
        else:
            self.opti.set_initial(self.var_q, np.zeros(self.reduced_robot.model.nq))
            self.opti.set_value(self.param_q_last, np.zeros(self.reduced_robot.model.nq))
            self.opti.set_value(self.param_alpha, 0.)

        target_Tf = np.eye(4)
        target_Tf[:3, 3] = target_translation
        target_quat = pin.Quaternion(target_quaternion[3], target_quaternion[0], target_quaternion[1], target_quaternion[2])
        target_Tf[:3, :3] = target_quat.matrix()
        self.opti.set_value(self.param_Tf, target_Tf)

        try:
            sol = self.opti.solve()

            q_solution = sol.value(self.var_q)

            return q_solution
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            return np.zeros([0,], dtype=np.float64)