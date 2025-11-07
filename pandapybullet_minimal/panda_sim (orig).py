import time
import math
import numpy as np
import pybullet as p


class PandaSim(object):

    def __init__(self, realtime=False):

        self.physics_client_id = None
        self.eye_in_hand_camera_projection_matrix = None
        self.prev_desired_position = None
        self.prev_actual_position = None

        self.robot_id = None
        self.setup_scene()

        self.realtime = realtime
        if self.realtime:
            p.setRealTimeSimulation(1)

    def setup_scene(self):

        print("Scene setup")

        # Connect with simulator
        self.physics_client_id = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setPhysicsEngineParameter(enableFileCaching=0)

        # Load scene
        import os.path as path
        module_path = path.dirname(path.abspath(__file__))
        self.scene_id = p.loadURDF(module_path + "/envs/plane/plane.urdf")
        p.setGravity(0, 0, -10.);
        print("Gravity set to: {}".format((0, 0, -10.)));

        # Load table
        table_urdf = module_path + "/envs/table/table.urdf"
        table_start_position = [0.35, 0.0, 0.0]
        table_start_orientation = [0.0, 0.0, 0.0]
        table_start_orientation_quat = p.getQuaternionFromEuler(table_start_orientation)
        self.table_id = p.loadURDF(table_urdf,
                              table_start_position,
                              table_start_orientation_quat,
                              flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
                              );
        print("Table Id: {}".format(self.table_id));

        # Load Panda robot
        robot_urdf = module_path + "/envs/frankaemika/robots/panda_arm_hand.urdf"
        robot_start_position = [0.0, 0.0, 0.88]
        robot_start_orientation = [0.0, 0.0, 0.0]
        robot_start_orientation_quat = p.getQuaternionFromEuler(robot_start_orientation)
        self.robot_id = p.loadURDF(
            robot_urdf,
            robot_start_position, robot_start_orientation_quat,
            useFixedBase=1,
            flags=p.URDF_USE_SELF_COLLISION  # |p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        )
        self.robotEndEffectorIndex = 8
        print("Robot id: {}".format(self.robot_id));

        # Set robot in initial pose
        # self.set_q((0.0, 0.0, 0.0, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi / 4.0))

    def set_q(self, joints):
        '''
        This sets the value of the robot joints.
        WARNING: This overrides the physics, do not use during simulation!!
        :param joints: tuple of size (7)
        :return:
        '''

        j1, j2, j3, j4, j5, j6, j7 = joints

        joint_angles = {}
        joint_angles["panda_joint_world"] = 0.0  # No actuation
        joint_angles["panda_joint1"] = j1
        joint_angles["panda_joint2"] = j2
        joint_angles["panda_joint3"] = j3
        joint_angles["panda_joint4"] = j4
        joint_angles["panda_joint5"] = j5
        joint_angles["panda_joint6"] = j6
        joint_angles["panda_joint7"] = j7
        joint_angles["panda_joint8"] = 0.0 # No actuation
        joint_angles["panda_hand_joint"] = 0.0 # No actuation
        joint_angles["panda_finger_joint1"] = 0.05 # No actuation
        joint_angles["panda_finger_joint2"] = 0.05 # No actuation
        joint_angles["camera_joint"] = 0.0  # No actuation
        joint_angles["camera_depth_joint"] = 0.0  # No actuation
        joint_angles["camera_depth_optical_joint"] = 0.0  # No actuation
        joint_angles["camera_left_ir_joint"] = 0.0  # No actuation
        joint_angles["camera_left_ir_optical_joint"] = 0.0  # No actuation
        joint_angles["camera_right_ir_joint"] = 0.0  # No actuation
        joint_angles["camera_right_ir_optical_joint"] = 0.0  # No actuation
        joint_angles["camera_color_joint"] = 0.0  # No actuation
        joint_angles["camera_color_optical_joint"] = 0.0  # No actuation

        for joint_index in range(p.getNumJoints(self.robot_id)):
            joint_name = p.getJointInfo(self.robot_id, joint_index)[1].decode('ascii')
            joint_angle = joint_angles.get(joint_name, 0.0)
            p.resetJointState(self.robot_id, joint_index, joint_angle)

    def set_gripper_state(self, opening):
        p.resetJointState(self.robot_id, 10, -opening/2)
        p.resetJointState(self.robot_id, 11, opening/2)

    def grab_eye_in_hand_camera_frame(self):
        fov = 60
        pixel_width = 320
        pixel_height = 200
        aspect = pixel_width / pixel_height
        near_plane = 0.01
        far_plane = 100
        if not self.eye_in_hand_camera_projection_matrix:
            self.eye_in_hand_camera_projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

        # Center of mass position and orientation (of link-20)
        pos, rot, _, _, _, _ = p.getLinkState(self.robot_id, linkIndex=20, computeForwardKinematics=True)

        rot_matrix = p.getMatrixFromQuaternion(rot)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, -1, 0)  # y-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        camera_eye_position = np.array(pos)
        camera_target_position = camera_eye_position + 0.2 * camera_vector

        view_matrix = p.computeViewMatrix(camera_eye_position, camera_target_position, up_vector)

        camera_imgs = p.getCameraImage(pixel_width, pixel_height,
                                       view_matrix, self.eye_in_hand_camera_projection_matrix,
                                       shadow=1, lightDirection=[1, 1, 1],
                                       renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_np_arr = (np.reshape(camera_imgs[2], (camera_imgs[1], camera_imgs[0], 4))) * (1. / 255.)
        depth_np_arr = np.reshape(camera_imgs[3], (camera_imgs[1], camera_imgs[0]))
        segmask_np_arr = np.reshape(camera_imgs[4], (camera_imgs[1], camera_imgs[0]))

        return rgb_np_arr, depth_np_arr, segmask_np_arr


    def get_qdq_J(self):
        qdq_matrix = np.array([np.array(
            p.getJointState(bodyUniqueId=self.robot_id, jointIndex=jointIndex)[:2]) for
            jointIndex in np.arange(1, 8)])
        q = qdq_matrix[:, 0]
        dq = qdq_matrix[:, 1]

        jac_t, jac_r = p.calculateJacobian(self.robot_id, self.robotEndEffectorIndex,
                                           [0., 0., 0.1034], list(q) + [0.] * 2, [0.] * 9,
                                           [0.] * 9)
        J = np.concatenate((np.array(jac_t)[:, :7], np.array(jac_r)[:, :7]), axis=0)
        J = np.reshape(J.flatten(order='F'), J.shape)

        return q, dq, J

    def get_x(self):
        robot_x = np.array(p.getLinkState(self.robot_id, self.robotEndEffectorIndex)[4])
        return robot_x

    def send_torque(self, tau):
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=np.arange(1, 8),
                                    controlMode=p.TORQUE_CONTROL, forces=tau)
        if not self.realtime:
            p.stepSimulation()

    def send_velocity(self, vel):
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=np.arange(1, 8),
                                    controlMode=p.VELOCITY_CONTROL, targetVelocities = vel)
        if not self.realtime:
            p.stepSimulation()

    def send_position(self, q):
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=np.arange(1, 8),
                                    controlMode=p.POSITION_CONTROL, targetPositions = q)
        if not self.realtime:
            p.stepSimulation()

    def send_cartesian_velocity(self, vel):
        desired_joints = self.get_x()+0.1*np.array(vel)
        desired_joints = p.calculateInverseKinematics(self.robot_id, 9, desired_joints, [1., 0., 0., 0.])[:7]
        q = self.get_qdq_J_pose()[0]
        qdot_desired = (np.array(desired_joints)-q)/0.1
        p.setJointMotorControlArray(self.robot_id, np.arange(1,8), p.VELOCITY_CONTROL, targetVelocities = qdot_desired)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--realtime", help="Whether or not to have the simulation running in realtime", required=False, default=False, type=bool)

    argument = parser.parse_args()
    if argument.realtime:
        print("Simulation running in realtime")

    panda_sim = PandaSim(argument.realtime)
