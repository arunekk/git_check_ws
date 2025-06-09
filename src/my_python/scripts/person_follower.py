#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import math
import time
import datetime
from std_srvs.srv import Empty
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import copy
import signal
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
import warnings
import os

#Pointcloud imports

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs

warnings.filterwarnings("ignore", category=RuntimeWarning)
if os.path.isfile("cmd_vel_log.txt"):
    os.remove("cmd_vel_log.txt")

lidar_points = []
max_width = 0.60
height = 1.27
vertex1 = [-max_width / 2, 0]
vertex2 = [max_width / 2, 0]
vertex3 = [0, height]

RFLAG = False
LFLAG = False
frontStopflag = False
suddenStopflag = False
leftStopflag = False
rightStopflag = False

## pointcloud variables
tf_buffer = None
tf_listener = None
camera_obstacle = False
ROBOT_RADIUS = 0.25 
SAFETY_DISTANCE = 0.5 
TOTAL_SAFE_DISTANCE = ROBOT_RADIUS + SAFETY_DISTANCE  


RUN = True
cmd_vel_topic = '/cmd_vel'
velocity_message = Twist()
velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
velocity_message.linear.y = 0
velocity_message.linear.z = 0
velocity_message.angular.x = 0
velocity_message.angular.y = 0
# rospy.set_param('/person_follower/tracker', "0")
retGoalValuesAng = [100, 0, 0, 0]

# k_a = 0.6
# k_d = 0.8
k_a = 0.6
k_d = 1.3
backStopflag = False
leftStopflag = False
rightStopflag = False
frontStopflag = False
suddenStopflag = False

RFLAG = False
LFLAG = False

botCurrentSpeed = 0
OdomVel = 0
rospy.set_param('/person_follower/movement', "0")
last_vel_x = 0

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log_file_name = "cmd_vel_log.txt"

hardcounter = 0

is_running_A = False
is_running_B = False
is_running_C = False
is_running_D = False


def write_log(message):
    log_file = open(log_file_name, "a")
    log_file.write(f"{message}\n")
    log_file.close()


def getbotSpeed(datain):
    global last_vel_x
    global is_running_D

    if is_running_D:
        return 0
    is_running_D = True

    global botCurrentSpeed
    global OdomVel
    botCurrentSpeed = datain.twist.twist.linear.x
    OdomVel = datain.twist.twist.linear.x

    PF_Run = rospy.get_param('/person_follower/person_follower_set_enable', False)
    if not PF_Run:
        last_vel_x = 0

    is_running_D = False
    return 0


def person_detect(data_):
    global k_d
    global k_a
    global botCurrentSpeed
    global backStopflag
    global leftStopflag
    global rightStopflag
    global frontStopflag

    global suddenStopflag
    global retGoalValuesAng
    global depth_image
    global last_vel_x
    global OdomVel
    global is_running_A
    global hardcounter
    global RFLAG
    global LFLAG

    global camera_obstacle

    if is_running_A:
        return 0
    is_running_A = True

    depth_, angle_, depvel_, angvel_ = [float(x) for x in data_.data.split(",")]
    obsflag = 0
    hardstopflag = 0
    if not depth_image is None:
        # print('camera')
        obsflag, hardstopflag = Obstacledetector(depth_image, depth_, angle_)

    if 0 < depth_ <= 0.35:
        depth_ = (0.35 - 0.6)
    else:
        depth_ = (depth_ - 0.6)

    if depth_ < 0.2:  # depth from origin(0.6m)
        if -3 < angle_ < 3:
            angle_ = 0

    elif 0.2 <= depth_:
        if -2 < angle_ < 2:
            angle_ = 0

    retGoalValuesAng = [depth_, angle_, depvel_, angvel_]
    velocity_message.linear.x = 0
    velocity_message.angular.z = 0
    frontStopflagObs = False
    depvel_ = retGoalValuesAng[2]

    if obsflag == 20:
        frontStopflagObs = True

    # print(rospy.get_param('/person_follower/tracker'))
    if rospy.get_param('/person_follower/tracker') == "0":

        if backStopflag:
            rospy.set_param('/person_follower/movement', "Back Obstacle detected")

        if LFLAG:
            rospy.set_param('/person_follower/movement', "Right Obstacle detected")

        if RFLAG:
            rospy.set_param('/person_follower/movement', "Left Obstacle detected")

        if frontStopflag and (depth_ > 0.5):
            rospy.set_param('/person_follower/movement', "Front Obstacle detected")

        if frontStopflagObs and (depth_ > 0.4):
            rospy.set_param('/person_follower/movement', "Front Obstacle detected")

        # if camera_obstacle:
        #     rospy.set_param('/person_follower/movement', "Lower Obstacle detected")

        # if hardstopflag :
        # rospy.set_param('/person_follower/movement',"*******************************")

        if not (backStopflag or RFLAG or LFLAG or frontStopflag or frontStopflagObs):
            rospy.set_param('/person_follower/movement', "0")
    if rospy.get_param('/person_follower/tracker') != "0":
        rospy.set_param('/person_follower/movement', "0")

    if retGoalValuesAng[0] <= -0.15:
        velocity_message.linear.x = 0
        assigned_speed = 0

        if (leftStopflag and retGoalValuesAng[1] < 0) or (rightStopflag and retGoalValuesAng[1] > 0):
            velocity_message.angular.z = 0
        else:
            if abs(retGoalValuesAng[1]) <= 22:
                velocity_message.angular.z = math.radians(k_a * -2 * retGoalValuesAng[1])

            else:

                velocity_message.angular.z = 1.7 * math.radians(k_a * -2.1 * retGoalValuesAng[1])

    elif -0.15 < retGoalValuesAng[0] <= 0.05:
        velocity_message.linear.x = 0
        assigned_speed = 0

        if (leftStopflag and retGoalValuesAng[1] < 0) or (rightStopflag and retGoalValuesAng[1] > 0):
            velocity_message.angular.z = 0
        else:
            if abs(retGoalValuesAng[1]) <= 24:
                velocity_message.angular.z = math.radians(k_a * -2 * retGoalValuesAng[1])

            else:

                velocity_message.angular.z = 1.7 * math.radians(k_a * -2.2 * retGoalValuesAng[1])

    elif 0.05 < retGoalValuesAng[0]:
        if retGoalValuesAng[0] != 100:

            if not (frontStopflag or frontStopflagObs):  # for linear movement
                decfac = 1
                if -0.7 <= depvel_ < -0.2:
                    decfac = 4
                elif -1 <= depvel_ < -0.7:
                    decfac = 6
                elif -1.4 <= depvel_ < -1:
                    decfac = 9
                elif depvel_ < -1.4:
                    decfac = 11
                ##########################
                if botCurrentSpeed <= (retGoalValuesAng[0] * 0.34):
                    if not (frontStopflag or frontStopflagObs):
                        angfac = 1
                        if abs(retGoalValuesAng[1]) > 13:  # or cntr>0:
                            angfac = 0.4
                        botCurrentSpeed = botCurrentSpeed + 0.3 * max(min(depvel_, 0.6), 0.2) * angfac
                        assigned_speed = max(0, botCurrentSpeed)

                    else:
                        assigned_speed = 0
                else:
                    if not (frontStopflag or frontStopflagObs):
                        assigned_speed = max(0, botCurrentSpeed - k_d * 0.017 * decfac)
                    else:
                        assigned_speed = 0

            else:
                assigned_speed = 0

        ##############################
        else:  # 100
            if not (frontStopflagObs or frontStopflag or backStopflag):
                # print("botCurrentSpeed",botCurrentSpeed)
                assigned_speed = botCurrentSpeed - k_d * 0.017 * 4

                if botCurrentSpeed < 0.04:
                    assigned_speed = 0

            else:
                assigned_speed = 0

        if OdomVel < 0.18 and retGoalValuesAng[0] <= 0.95:  # for angle movement
            if (leftStopflag and retGoalValuesAng[1] < 0) or (rightStopflag and retGoalValuesAng[1] > 0):
                velocity_message.angular.z = 0

            else:
                if retGoalValuesAng[0] <= 0.5:
                    k_c = 1.7

                elif 0.5 < retGoalValuesAng[0] <= 1.1:
                    k_c = 1.3
                else:
                    k_c = 1.1
                if abs(retGoalValuesAng[1]) <= 23:
                    velocity_message.angular.z = math.radians(k_a * -1.65 * retGoalValuesAng[1])

                else:
                    # print("k_c       ",k_c)
                    velocity_message.angular.z = k_c * math.radians(k_a * -1.5 * retGoalValuesAng[1])

                # print(retGoalValuesAng[1],velocity_message.angular.z)
        else:
            if (retGoalValuesAng[0]) <= 1:

                if (leftStopflag and retGoalValuesAng[1] < 0) or (rightStopflag and retGoalValuesAng[1] > 0):
                    velocity_message.angular.z = 0
                else:
                    velocity_message.angular.z = math.radians(k_a * -1.55 * retGoalValuesAng[1])

            else:

                if (leftStopflag and retGoalValuesAng[1] < 0) or (rightStopflag and retGoalValuesAng[1] > 0):
                    velocity_message.angular.z = 0
                else:
                    velocity_message.angular.z = math.radians(k_a * -1.2 * retGoalValuesAng[1])

    stopflag = "     "
    if assigned_speed == 0:
        stopflag = "  b   "
        # print("    STOP")
        if last_vel_x < 0.04:
            assigned_speed = 0
        elif 0.04 <= last_vel_x < 0.2:
            assigned_speed = last_vel_x - 0.05
        elif 0.2 <= last_vel_x < 0.35:
            assigned_speed = last_vel_x - 0.05
        elif 0.35 <= last_vel_x < 0.5:
            assigned_speed = last_vel_x - 0.06
        else:
            assigned_speed = last_vel_x - 0.07

    velocity_message.linear.x = max(0, min(assigned_speed, 0.68))
    velocity_message.linear.x = 0.5 * velocity_message.linear.x + 0.5 * last_vel_x
    # print(round(velocity_message.linear.x,2))
    if suddenStopflag or hardstopflag:
        velocity_message.linear.x = 0
    last_vel_x = velocity_message.linear.x
    write_log(str(round(velocity_message.linear.x, 3)) + "," + str(stopflag))
    velocity_publisher.publish(velocity_message)

    is_running_A = False
    return 0


def handler(signum, frame):
    global RUN
    RUN = False


def doAction():
    global RUN
    global retGoalValuesAng
    signal.signal(signal.SIGINT, handler)
    while RUN:
        # pass
        if len(retGoalValuesAng) == 0:
            print('waiting.', end='\r')
            continue
        if retGoalValuesAng[0] == 0 and retGoalValuesAng[1] == 0:
            print('Idle')

    print('       --    ')
    print('      ----   ')
    print('    - o o -  ')
    print('  !!Bye Bye!!')
    print('     - O -  ')
    print('      --    ')
    print('       -     ')


bridge = CvBridge()
depth_image = None


def image_depth_callback(msg):
    global is_running_B

    if is_running_B:
        return 0
    is_running_B = True

    global depth_image
    cv2_img1 = bridge.imgmsg_to_cv2(msg, "passthrough")
    depth_image = np.array(cv2_img1, dtype=np.float32)

    is_running_B = False
    return 0


def Obstacledetector(ffulldepthframe, depthoftarget, angle):
    ffulldepthframe = ffulldepthframe.astype('uint16')
    fulldepthframe = ffulldepthframe.copy()
    nfulldepthframe = ffulldepthframe.copy()
    fulldepthframe = fulldepthframe[::2, ::2]
    nfulldepthframe = nfulldepthframe[::2, ::2]

    obstacleflag = 0
    hardobstacleflag = 0

    limit = depthoftarget * 1000

    ki = 1
    if abs(angle) > 20:
        ki = 0.8

    idxOfnonzero_ = np.where(fulldepthframe > 0)[0]
    nidxOfnonzero_ = np.where(nfulldepthframe > 0)[0]

    ratio = idxOfnonzero_.shape[0] / (fulldepthframe.size + 1)

    depframe = fulldepthframe.copy()
    depframe[depframe == 0] = 10000

    boundarray = np.zeros((240, 320))
    boundarray[235:240, 128:192] = min(1400, limit * ki)  # 1down
    boundarray[220:235, 128:192] = min(1400, limit * 0.9 * ki)  # 1up
    boundarray[200:220, 128:192] = min(1600, limit * 0.95 * ki)  # 2
    boundarray[180:200, 128:192] = min(1600, limit * 0.9 * ki)  # 3
    boundarray[160:180, 128:192] = min(1500, limit * 0.7 * ki)  # 4
    boundarray[0:160, 128:192] = min(1000, limit * 0.7)  # 5

    boundarray[:200, 96:128] = min(1000, limit * 0.7)  # 6
    boundarray[:200, 96:112] = min(1000, limit * 0.7)  # 7
    boundarray[400:, 96:128] = min(800, limit)  # 12
    boundarray[400:, 192:224] = min(800, limit)  # 13

    boundarray[-60:, 64:96] = min(600, limit)  # 8
    boundarray[-60:, 224:256] = min(600, limit)  # 9
    boundarray[:-60, 64:96] = min(680, limit * 0.9)  # 10
    boundarray[:-60, 224:256] = min(680, limit * 0.9)  # 11

    diff = depframe - boundarray

    noofpts = np.where(diff < 0)[0].shape[0]

    if noofpts > 150 or ratio < 0.8:
        obstacleflag = 20
    # print("Obstacle detected")
    if ratio < 0.8:
        hardobstacleflag = 20

    return obstacleflag, hardobstacleflag


LocalObj = np.ones((1, 1))
degree_increment = 0


def lidar_callback(msg):
    global is_running_C, LocalObj, degree_increment

    if is_running_C:
        return 0
    is_running_C = True

    LocalObj = np.array(msg.ranges)
    LocalObj[np.isinf(LocalObj)] = 10
    degree_increment = 180 / len(msg.ranges)
    is_running_C = False
    return 0

def pointcloud_callback(msg):
    global camera_obstacle
    
    try:
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        
        if not points:
            camera_obstacle = False
            return
        
        points = np.array(points)
        min_distance = float('inf')
        obstacle_count = 0
        low_obstacle_count = 0  # Count obstacles below LiDAR level
        
        # Transform points to base_link frame if needed
        if msg.header.frame_id != "base_link":
            try:
                # Get transform from point cloud frame to base_link
                transform = tf_buffer.lookup_transform("base_link", msg.header.frame_id, 
                                                     msg.header.stamp, rospy.Duration(0.1))
                
                # Apply transformation to each point
                transformed_points = []
                for point in points:
                    point_stamped = PointStamped()
                    point_stamped.header.frame_id = msg.header.frame_id
                    point_stamped.point.x = point[0]
                    point_stamped.point.y = point[1]
                    point_stamped.point.z = point[2]
                    
                    transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
                    transformed_points.append([transformed_point.point.x, 
                                             transformed_point.point.y, 
                                             transformed_point.point.z])
                
                points = np.array(transformed_points)
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                   tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed: {e}")
                # Use points as-is if transform fails
        
        # Check each point for obstacles
        for point in points:
            x, y, z = point
            
            # Filter points by height - focus on obstacles that LiDAR cannot see well
            # LiDAR is at 0.245m height, so focus on obstacles below AND above this level
            # Remove only obvious ground/noise points but keep very low obstacles
            if z < -0.1 or z > 2.5:  # Allow very low points (8cm = 0.08m should pass)
                continue
            
            # Filter out points that are too far away to be relevant
            if abs(x) > 3.0 or abs(y) > 3.0:
                continue
            
            # Calculate horizontal distance from robot center
            horizontal_distance = math.sqrt(x**2 + y**2)
            
            # Check if point is within safety zone
            if horizontal_distance < TOTAL_SAFE_DISTANCE:
                min_distance = min(min_distance, horizontal_distance)
                obstacle_count += 1
                
                # Count low obstacles (below LiDAR level)
                if z < 0.25:  # Below LiDAR level (0.245m + small margin)
                    low_obstacle_count += 1
                
                # Debug: Print obstacle points with height info
                if obstacle_count <= 10:  # Print more detections for debugging
                    height_status = "LOW (LiDAR blind)" if z < 0.25 else "HIGH (LiDAR visible)"
                    rospy.loginfo(f"Obstacle point {obstacle_count}: x={x:.2f}, y={y:.2f}, z={z:.2f}m, dist={horizontal_distance:.2f}m [{height_status}]")
                    # Add this in the pointcloud_callback function, right after the height filtering
                    if 0.05 < z < 0.15 and horizontal_distance < 1.0:  # Focus on low obstacles nearby
                        rospy.loginfo(f"DEBUG: Low obstacle candidate: x={x:.3f}, y={y:.3f}, z={z:.3f}m (8cm = 0.080m), dist={horizontal_distance:.3f}m")
        
        # Update obstacle status - be more sensitive for safety
        # Reduced threshold: even 1 point within safety zone should trigger stopping
        if obstacle_count >= 2 and min_distance < TOTAL_SAFE_DISTANCE:
            camera_obstacle = True
            if low_obstacle_count > 0:
                rospy.logwarn(f"Camera: LOW obstacle detected at {min_distance:.2f}m (LiDAR blind spot) - {low_obstacle_count} low points, {obstacle_count} total points")
            else:
                rospy.logwarn(f"Camera: Obstacle detected at {min_distance:.2f}m from robot center ({obstacle_count} points)")
        else:
            camera_obstacle = False
            if obstacle_count > 0:
                rospy.loginfo(f"Camera: Found {obstacle_count} points but distance > {TOTAL_SAFE_DISTANCE:.2f}m (safe)")
            
    except Exception as e:
        rospy.logerr(f"Error processing point cloud data: {e}")
        camera_obstacle = True  # Fail-safe: assume obstacle if error


def is_point_in_triangle(point, v1, v2, v3):
    def RobotPathwayZone(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    side1 = RobotPathwayZone(point, v1, v2)
    side2 = RobotPathwayZone(point, v2, v3)
    side3 = RobotPathwayZone(point, v3, v1)

    has_negative = (side1 < 0) or (side2 < 0) or (side3 < 0)
    has_positive = (side1 > 0) or (side2 > 0) or (side3 > 0)

    if not (has_negative and has_positive):
        if 0.05 <= point[1] <= 0.5:
            return True, 'Critical Zone'
        """elif 0.75 < point[1] <= 0.9:
            return True, 'Less Critical Zone'
        elif 0.9 < point[1] <= 1.1:
            return True, 'Least Critical Zone'
        elif 1.1 < point[1] <= v3[1]:
            return True, 'Safe Zone'"""

    return False, None


def readLaser(move_dir=True):
    global leftStopflag, LocalObj, degree_increment
    global rightStopflag
    global frontStopflag
    global suddenStopflag
    global retGoalValuesAng
    global RFLAG
    global LFLAG
    global camera_obstacle

    ctCopyLidar = copy.deepcopy(LocalObj)

    if camera_obstacle:
        rospy.loginfo("ROBOT STOPPED: obstacle detected")
        suddenStopflag = True
        return
    
    critical_zone_count = 0

    if not camera_obstacle:
        for i in range(len(ctCopyLidar)):
            theta = i * degree_increment
            x = ctCopyLidar[i] * math.cos(math.radians(theta))
            y = ctCopyLidar[i] * math.sin(math.radians(theta))
            point = np.array([x, y])
            in_triangle, zone = is_point_in_triangle(point, vertex1, vertex2, vertex3)
            if in_triangle:
                #print(f"Point {point} is in {zone}")

                if zone in ['Critical Zone', 'Less Critical Zone', 'Least Critical Zone', 'Safe Zone']:
                    critical_zone_count += 1
                    if critical_zone_count > 3:
                        rospy.loginfo("ROBOT STOP")
                        suddenStopflag = True
                        break

    if critical_zone_count <=3 and not camera_obstacle:
        suddenStopflag = False


    rightValues = ctCopyLidar[:150]
    leftValues = ctCopyLidar[-150:]

    locOfobjRight = np.where(rightValues < 0.01)[0]
    locOfobjLeft = np.where(leftValues < 0.01)[0]

    '''if locOfobjRight.shape[0] > 2:
        print("right_obs")
        RFLAG = True
        rightStopflag = True

    else:
        RFLAG = False
        rightStopflag = False

    if locOfobjLeft.shape[0] > 2:
        print("left_obs")
        LFLAG = True
        leftStopflag = True

    else:
        RFLAG = False
        rightStopflag = False'''


if __name__ == '__main__':
    try:
        rospy.init_node('person_follower_node', anonymous=True)

        # Initialize TF buffer and listener and Pointcloud subscriber
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        rospy.Subscriber('/cloud_concatenated', PointCloud2, pointcloud_callback, queue_size=1)

        rospy.Subscriber("/person_position", String, person_detect)
        rospy.Subscriber('/scan', LaserScan, lidar_callback)
        rospy.Subscriber('/odom', Odometry, getbotSpeed)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, image_depth_callback)
        while not rospy.is_shutdown():
            readLaser()
            time.sleep(0.1)

    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")