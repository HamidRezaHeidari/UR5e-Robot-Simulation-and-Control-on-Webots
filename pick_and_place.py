# !pip install ultralytics
# !pip install onnxruntime

import sys
webot_path = 'C:\Program Files\Webots\lib\controller\python'
sys.path.append(webot_path)

import numpy as np
from ultralytics import YOLO

from controller import Robot, VacuumGripper, Motor, Camera, RangeFinder

# create the Robot instance
robot = Robot()

TIME_STEP = 64

camera = Camera('camera')
range_finder = RangeFinder('range-finder')

range_finder.enable(TIME_STEP)
camera.enable(TIME_STEP)

# import trained YOLOv8 Model ( webot_objects_detection.onnx file should be in directory)
trained_model = YOLO("webot_objects_detection.onnx", task="detect")


# grab RGB frame and convert to numpy ndarray
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

    def inverse_kinematic(self, x, y, z):
        return


## robot instance

ur5 = UR5e()

a_base = [0, 0, 0, 0, 0, 0]
ur5.set_arm_pos(a_base)
ur5.set_gripper_pos(state='on')

robot.step(TIME_STEP)
img = get_rgb_frame()

# a_final = [-1.2, 2.9, 0.8, 1.97, 1.57, 0]
a_final = [-1.3, 2, 1, 2, 1.1, 0]

def pick_and_place(d):
    d_label, d_x, d_y = d

    labels = {"Biscuit Box": 0, "Box": 1, "Can": 2, "Mouse": 3, "Phone": 4}

    h_basket = 0.5+0.1     # height of table +  height of basket
    h_object = 0.5+0.015   # height of table + minimum height of objects(phone)
    h_gripper = 0.13

    d_z_basket = h_gripper + h_basket
    d_z_obj =  h_gripper + h_object

    red_basket = [0.6, -0.2, d_z_basket] # mouse and phone
    green_basket = [0.6, 0.2, d_z_basket]  # biscuit and can
    blue_basket = [0.01, 0.6282, 0.152+h_gripper]  # box

    if d_label == labels[3] or d_label == labels[4]:
        final_d_x, final_d_y, final_d_z = red_basket
    elif d_label == labels[0] or d_label == labels[2]:
        final_d_x, final_d_y, final_d_z = green_basket
    else:
        final_d_x, final_d_y, final_d_z = blue_basket

    a = 0
    while a!=1:
        ur5.inverse_kinematic(d_x, d_y, d_z_obj)
        if (ur5.get_EE_position()[0] - d_x < 0.0001) and (ur5.get_EE_position()[2] - d_z_obj < 0.0001):
            ur5.set_gripper_pos(state='on')  # Turn On Gripper
            while a!=1:
                ur5.inverse_kinematic(final_d_x, final_d_y, final_d_z)
                if (ur5.get_EE_position()[0] - final_d_x < 0.0001) and (ur5.get_EE_position()[2] - final_d_z < 0.0001):
                    ur5.set_gripper_pos(state='off')  # Turn Off Gripper
                    a = 1


while robot.step(TIME_STEP) != -1:

    ur5.set_arm_pos(a_final)
    img = get_rgb_frame()
    camera.saveImage("pic.jpeg", 80)

    if (ur5.get_arm_pos()[0] - a_final[0] < 0.0001) and (ur5.get_arm_pos()[1] - a_final[1] < 0.0001):
        objects = trained_model.predict(source="pic.jpeg", save=True, save_txt=True)

        boxes = objects[0].boxes
        objects_bb_data = []

        for box in boxes:
            cls = np.array(box.cls)
            bb = np.array(box.xywhn)
            bb_data = np.append(cls, bb)

            objects_bb_data.append(bb_data)

        print("objects_data :", objects_bb_data)

        X_a, Y_a = -0.150562, -0.21049
        X_b, Y_b = -1.25122, 0.377894

        real_data = []
        for i in range(len(objects_bb_data)):

            x_real = X_a + (X_b-X_a)*(1-objects_bb_data[i][2])
            y_real = Y_a + (Y_b-Y_a)*(objects_bb_data[i][1])

            real_data.append([objects_bb_data[i][0], x_real, y_real])

        print("\n", "real object data :", real_data)

        for i in range(len(real_data)):
            pick_and_place(real_data[i])

        break



