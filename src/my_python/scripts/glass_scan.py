#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan

def scan_callback(msg):
    intensities = msg.intensities
    count = 0
    glass_detected = False

    for intensity in intensities:
        if intensity == 0.0:
            count += 1
            if count >= 5:
                rospy.loginfo("Glass detected")
                glass_detected = True
                break
        else:
            count = 0

    if not glass_detected:
        rospy.loginfo("Solid obstacle")
        print("Solid obstacle detected")

def main():
    rospy.init_node('glass_detector', anonymous=True)
    rospy.Subscriber('/scan', LaserScan, scan_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
