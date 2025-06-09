#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import time

z_data = []

def pointcloud_callback(msg):
    global z_data
    points = pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)
    points_array = np.array(list(points))

    if points_array.size > 0:
        z_mean = np.mean(points_array[:, 2])  # Mean Z-distance
        z_data.append(z_mean)

def detect_object_fft(z_data, fs=10, low_freq_limit=1.0, threshold=0.05):
    L = len(z_data)
    Y = np.fft.fft(z_data)
    f = np.fft.fftfreq(L, d=1/fs)

    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    f_positive = f[:L // 2 + 1]

    # Sum low-frequency energy below limit
    low_freq_energy = np.sum(P1[f_positive <= low_freq_limit])
    print(f"Low-frequency energy: {low_freq_energy:.4f}")

    return low_freq_energy > threshold

def main():
    global z_data

    fs = 10         # Sampling frequency (Hz)
    duration = 10   # Duration to collect data (seconds)
    expected_samples = int(duration * fs)

    rospy.init_node('fft_object_detector', anonymous=True)
    rospy.Subscriber('/camera_03/depth/points', PointCloud2, pointcloud_callback)

    rospy.loginfo("⏳ Collecting pointcloud data for 10 seconds...")
    start = time.time()
    while not rospy.is_shutdown() and time.time() - start < duration:
        rospy.sleep(0.1)

    # 🔒 Lock z_data to prevent async modifications
    z_data_snapshot = z_data.copy()
    actual_samples = len(z_data_snapshot)

    if actual_samples < 2:
        rospy.logwarn("⚠️ Not enough data collected.")
        return

    if actual_samples < expected_samples * 0.8:
        rospy.logwarn(f"⚠️ Collected only {actual_samples} samples, expected around {expected_samples}.")

    # Time axis
    t = np.arange(actual_samples) / fs
    z_data_snapshot = z_data_snapshot[:actual_samples]

    # --- Plot Time Domain Signal ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, z_data_snapshot)
    plt.title("Z-Mean Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Z (m)")
    plt.grid(True)

    # --- FFT ---
    L = len(z_data_snapshot)
    Y = np.fft.fft(z_data_snapshot)
    f = np.fft.fftfreq(L, d=1/fs)
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]

    # --- Plot Frequency Domain ---
    plt.subplot(1, 2, 2)
    plt.plot(f[:L // 2 + 1], P1)
    plt.title("FFT of Z-Mean")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- Decision Logic ---
    object_present = detect_object_fft(z_data_snapshot, fs=fs, low_freq_limit=1.0, threshold=0.05)
    print("✅ 📦 Object Detected" if object_present else "❌ 🌫️ No Object (Noise Dominant)")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
