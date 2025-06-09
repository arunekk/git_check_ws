#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import time

metric_data = []

def pointcloud_callback(msg):
    global metric_data
    points = pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)
    points_array = np.array(list(points))

    if points_array.size > 0:
        # --- METRIC 1: Mean Euclidean distance from origin ---
        distances = np.linalg.norm(points_array[:, :3], axis=1)
        mean_distance = np.mean(distances)
        metric_data.append(mean_distance)

        # --- Other options (comment/uncomment as needed) ---
        # mean_z = np.mean(points_array[:, 2])          # Just Z-axis
        # density = len(points_array)                   # Number of points (could represent object density)
        # bounding_box_volume = np.prod(np.ptp(points_array, axis=0))  # Volume covered by points
        # metric_data.append(mean_z)  # for example

def compute_fft(signal, fs):
    L = len(signal)
    Y = np.fft.fft(signal)
    f = np.fft.fftfreq(L, d=1/fs)
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    return f[:L // 2 + 1], P1

def main():
    global metric_data

    rospy.init_node('pointcloud_fft_analyzer', anonymous=True)
    rospy.Subscriber('/cloud_concatenated', PointCloud2, pointcloud_callback)

    fs = 10         # Sampling frequency (Hz)
    duration = 10   # Duration in seconds
    expected_samples = fs * duration

    rospy.loginfo("⏳ Collecting pointcloud data for %d seconds...", duration)
    start = time.time()
    while not rospy.is_shutdown() and time.time() - start < duration:
        rospy.sleep(0.1)

    # Freeze the data
    data_snapshot = metric_data.copy()
    actual_samples = len(data_snapshot)

    if actual_samples < 2:
        rospy.logwarn("⚠️ Not enough data collected.")
        return

    rospy.loginfo("✅ Collected %d samples", actual_samples)

    t = np.arange(actual_samples) / fs
    data_snapshot = data_snapshot[:actual_samples]

    # --- Plot Time-Domain Signal ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, data_snapshot, marker='o', linestyle='-')
    plt.title("Mean Euclidean Distance Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Distance (m)")
    plt.grid(True)

    # --- FFT ---
    f_axis, magnitude = compute_fft(data_snapshot, fs)

    # --- Plot Frequency-Domain Signal ---
    plt.subplot(1, 2, 2)
    plt.plot(f_axis, magnitude)
    plt.title("FFT of Mean Distance")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
