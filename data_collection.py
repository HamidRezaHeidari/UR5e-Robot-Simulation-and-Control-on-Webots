import sys
webot_path = 'C:\Program Files\Webots\lib\controller\python'
sys.path.append(webot_path)

import numpy as np
from controller import Robot, VacuumGripper, Motor, Camera, RangeFinder

# create the Robot instance
robot = Robot()

TIME_STEP = 64

camera = Camera('camera')
range_finder = RangeFinder('range-finder')

range_finder.enable(TIME_STEP)
camera.enable(TIME_STEP)


## grab RGB frame and convert to numpy ndarray
def get_rgb_frame() -> np.ndarray:
    image_array = camera.getImageArray()
    np_image = np.array(image_array, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 3))
    return np_image


# grab Depth frame and convert to numpy ndarray
def get_depth_frame() -> np.ndarray:
    image_array = range_finder.getImageArray()
    np_image = np.array(image_array, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 3))
    return image_array


class UR5e:
    def __init__(self, name="my_robot"):
        # get the motor devices
        m1 = robot.getDevice('shoulder_lift_joint')
        m2 = robot.getDevice('shoulder_pan_joint')
        m3 = robot.getDevice('elbow_joint')

        m4 = robot.getDevice('wrist_1_joint')
        m5 = robot.getDevice('wrist_2_joint')
        m6 = robot.getDevice('wrist_3_joint')

        self.vac = robot.getDevice('vacuum gripper')
        self.vac.enablePresence(1)

        self.motors_list = [m1, m2, m3, m4, m5, m6]

        self.gps = robot.getDevice('gps')
        self.gps.enable(1)

        sampling_period = 1

        for m in self.motors_list:
            m.getPositionSensor().enable(sampling_period)
            m.enableForceFeedback(sampling_period)
            m.enableTorqueFeedback(sampling_period)

    def set_arm_torques(self, torques):
        for i, motor in enumerate(self.motors_list):
            motor.setTorque(torques[i])

    def set_gripper_pos(self, state='on'):
        ''' state : set vacuum gripper "on" or "off" for vacuum activation'''
        if state == 'on' or state == 'On' or state == 'ON':
            self.vac.turnOn()
        else:
            self.vac.turnOff()

    def set_arm_pos(self, pos):
        for i, motor in enumerate(self.motors_list):
            motor.setPosition(pos[i])

    def get_arm_pos(self):
        p = [m.getPositionSensor().getValue() for m in self.motors_list]
        return p

    def get_gripper_pos(self):
        p = [m.getPositionSensor().getValue() for m in self.gripper_list]
        return p

    def get_EE_position(self):
        return self.gps.value


## robot instance
ur5 = UR5e()

a_base = [0, 0, 0, 0, 0, 0]
ur5.set_arm_pos(a_base)
ur5.set_gripper_pos(state='on')

robot.step(TIME_STEP)

img = get_rgb_frame()
# depth = get_depth_frame()

a1 = [-1.3, 2, 1, 2, 1.1, 0]

a2 = [-1.3, 2, 1, 2, 1.2, 0]

a3 = [-1.3, 1.65, 1.52, -1.2, -1.08, 0]

a4 = [-0.9, 1.8, 1.1, -1, -1.08, 0]

a = [a1, a2, a3, a4]

k = 4 #set manually from 1 to 4

for i in range(4):
    for j in range(6):
        ur5.set_arm_pos(a_base)
        random = 0.1 * np.random.random(6) - 0.01

        while robot.step(TIME_STEP) != -1:
            a_final = a[i]+random
            ur5.set_arm_pos(a_final)
            img = get_rgb_frame()

            camera.saveImage(f"Pic{i}{j}{k}.jpeg",80)

            if (ur5.get_arm_pos()[0] - a_final[0] < 0.0001) and (ur5.get_arm_pos()[1] - a_final[1] < 0.0001):
                break
            # depth = get_depth_frame()


