#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Twist
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import math

# Global variables
cmd_vel_pub = None
obstacle_detected = False
tf_buffer = None
tf_listener = None
lidar_obstacle = False
camera_obstacle = False

# Robot parameters
ROBOT_RADIUS = 0.25  # 25cm radius from URDF
SAFETY_DISTANCE = 0.5  # 50cm safety distance
TOTAL_SAFE_DISTANCE = ROBOT_RADIUS + SAFETY_DISTANCE  # 75cm total

def laser_callback(msg):
    """Process LiDAR data and check for obstacles"""
    global lidar_obstacle
    
    try:
        # Check each laser range reading
        min_distance = float('inf')
        angle = msg.angle_min
        
        for i, range_val in enumerate(msg.ranges):
            # Skip invalid readings
            if math.isnan(range_val) or math.isinf(range_val) or range_val < msg.range_min or range_val > msg.range_max:
                angle += msg.angle_increment
                continue
            
            # Convert polar to cartesian (relative to laser frame)
            x = range_val * math.cos(angle)
            y = range_val * math.sin(angle)
            
            # Transform from laser frame to base_link frame
            # Laser is at (0.213, 0, 0.16) relative to base_link
            x_base = x + 0.213
            y_base = y + 0.0
            
            # Calculate distance from robot center to obstacle
            distance_from_center = math.sqrt(x_base**2 + y_base**2)
            
            # Check if obstacle is within safety zone
            if distance_from_center < TOTAL_SAFE_DISTANCE:
                min_distance = min(min_distance, distance_from_center)
            
            angle += msg.angle_increment
        
        # Update obstacle status
        if min_distance < TOTAL_SAFE_DISTANCE:
            lidar_obstacle = True
            rospy.logwarn(f"LiDAR: Obstacle detected at {min_distance:.2f}m from robot center")
        else:
            lidar_obstacle = False
            
    except Exception as e:
        rospy.logerr(f"Error processing laser data: {e}")
        lidar_obstacle = True  # Fail-safe: assume obstacle if error

def pointcloud_callback(msg):
    """Process concatenated point cloud data and check for obstacles"""
    global camera_obstacle
    
    try:
        # Extract points from point cloud
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        
        if not points:
            camera_obstacle = False
            return
        
        points = np.array(points)
        min_distance = float('inf')
        obstacle_count = 0
        
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
            
            # Filter points by height - focus on obstacles that LiDAR cannot see
            # LiDAR is at 0.245m height, so focus on obstacles below this level
            # Also include some obstacles above LiDAR level for comprehensive detection
            if z < 0.02 or z > 2.0:  # Very low points might be noise, very high points irrelevant
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
                
                # Debug: Print obstacle points with height info
                if obstacle_count <= 5:  # Print first few detections
                    height_status = "LOW (LiDAR blind)" if z < 0.24 else "HIGH (LiDAR visible)"
                    rospy.loginfo(f"Obstacle point {obstacle_count}: x={x:.2f}, y={y:.2f}, z={z:.2f}, dist={horizontal_distance:.2f} [{height_status}]")
        
        # Update obstacle status - be more sensitive since these might be LiDAR blind spots
        if obstacle_count >= 3 and min_distance < TOTAL_SAFE_DISTANCE:  # Even 1 point is enough for safety
            camera_obstacle = True
            low_obstacle_count = sum(1 for point in points if len(point) > 2 and 0.02 <= point[2] < 0.24 and math.sqrt(point[0]**2 + point[1]**2) < TOTAL_SAFE_DISTANCE)
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

def publish_safe_cmd_vel():
    """Publish velocity command: move forward at 0.1 m/s if safe, stop if obstacles detected"""
    global obstacle_detected
    
    # Update global obstacle status
    obstacle_detected = lidar_obstacle or camera_obstacle
    
    cmd = Twist()
    
    if obstacle_detected:
        # Stop the robot
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0
        
        if lidar_obstacle and camera_obstacle:
            rospy.logwarn_throttle(1, "ROBOT STOPPED: Obstacles detected by both LiDAR and Camera!")
        elif lidar_obstacle:
            rospy.logwarn_throttle(1, "ROBOT STOPPED: Obstacle detected by LiDAR!")
        elif camera_obstacle:
            rospy.logwarn_throttle(1, "ROBOT STOPPED: Obstacle detected by Camera!")
    else:
        # Move forward at 0.1 m/s
        cmd.linear.x = 0.1
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0
        rospy.loginfo_throttle(5, "Path clear - Moving forward at 0.1 m/s")
    
    cmd_vel_pub.publish(cmd)

def main():
    """Main function to initialize node and start obstacle avoidance"""
    global cmd_vel_pub, tf_buffer, tf_listener
    
    # Initialize ROS node
    rospy.init_node('obstacle_avoidance_node', anonymous=True)
    rospy.loginfo("Starting Obstacle Avoidance Node...")
    
    # Initialize TF buffer and listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    # Publishers
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    
    # Subscribers
    laser_sub = rospy.Subscriber('/scan', LaserScan, laser_callback, queue_size=1)
    pointcloud_sub = rospy.Subscriber('/cloud_concatenated', PointCloud2, pointcloud_callback, queue_size=1)
    
    rospy.loginfo("Subscribed to:")
    rospy.loginfo("  - /scan (LaserScan)")
    rospy.loginfo("  - /cloud_concatenated (PointCloud2)")
    rospy.loginfo("Publishing to:")
    rospy.loginfo("  - /cmd_vel (Twist)")
    rospy.loginfo("Robot behavior:")
    rospy.loginfo("  - Move forward at 0.1 m/s when path is clear")
    rospy.loginfo("  - Stop when obstacles detected within safety zone")
    rospy.loginfo("  - LiDAR covers obstacles at 0.245m height")
    rospy.loginfo("  - Depth cameras cover LOW obstacles (<0.24m) that LiDAR misses")
    rospy.loginfo(f"Safety parameters:")
    rospy.loginfo(f"  - Robot radius: {ROBOT_RADIUS}m")
    rospy.loginfo(f"  - Safety distance: {SAFETY_DISTANCE}m")
    rospy.loginfo(f"  - Total safe distance: {TOTAL_SAFE_DISTANCE}m")

    cmd = Twist()
    
    # Main control loop
    rate = rospy.Rate(20)  # 20 Hz
    
    while not rospy.is_shutdown():
        try:
            publish_safe_cmd_vel()
            rate.sleep()
        except rospy.ROSInterruptException:
            break
        except Exception as e:
            rospy.logerr(f"Error in main loop: {e}")
            # Fail-safe: publish stop command
            stop_cmd = Twist()
            cmd_vel_pub.publish(stop_cmd)
    
    cmd.linear.x = 0.0
    cmd_vel_pub.publish(cmd)
    rospy.loginfo("Obstacle Avoidance Node shutting down...")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        # Final fail-safe
        if cmd_vel_pub:
            stop_cmd = Twist()
            cmd_vel_pub.publish(stop_cmd)