import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# df = pd.read_csv('laser_values.csv', header=None)
df = pd.read_csv('camera_values.csv', header=None)
ranges = df.iloc[0].astype(float).values


angle_min = -np.pi / 2
angle_max = np.pi / 2
num_readings = len(ranges)
angle_increment = (angle_max - angle_min) / num_readings

angles = angle_min + np.arange(num_readings) * angle_increment
x_vals = ranges * np.cos(angles)
y_vals = ranges * np.sin(angles)


points = pd.DataFrame({'x': y_vals, 'y': x_vals, 'range': ranges})

V1 = np.array([-0.3, 0])
V2 = np.array([0.3, 0])
V3 = np.array([0, 1.27])

def is_inside_triangle(p, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)


xmin, ymin = np.min([V1, V2, V3], axis=0) - 0.1
xmax, ymax = np.max([V1, V2, V3], axis=0) + 0.1

grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T


inside_mask = np.array([is_inside_triangle(p, V1, V2, V3) for p in grid_points])
inside_points = grid_points[inside_mask]


distances = np.linalg.norm(inside_points, axis=1)

def zone_for_distance(d):
    if d <= 0.5:
        return 0 
    elif d <= 0.9:
        return 1  
    elif d <= 1.1:
        return 2  
    else:
        return 3  

zones_grid = np.array([zone_for_distance(d) for d in distances])


zone_colors_fill = ['#ff9999', '#ffcc99', '#ffff99', '#99ff99']

zone_colors_points = ['#e60000', '#e67300', '#e6e600', '#33cc33']

plt.figure(figsize=(10, 10))


plt.scatter(inside_points[:,0], inside_points[:,1], 
            c=[zone_colors_fill[z] for z in zones_grid], 
            s=6, marker='s', alpha=0.5, linewidth=0)


triangle = np.array([V1, V2, V3, V1])
plt.plot(triangle[:, 0], triangle[:, 1], 'k--', linewidth=1.5, label='Triangle Zone')


points['inside_triangle'] = [is_inside_triangle(np.array([row['x'], row['y']]), V1, V2, V3) for _, row in points.iterrows()]


def classify_zone(row):
    if row['inside_triangle']:
        if row['range'] <= 0.5:
            return 'Critical'
        elif row['range'] <= 0.9:
            return 'Less Critical'
        elif row['range'] <= 1.1:
            return 'Least Critical'
        else:
            return 'Safe'
    else:
        return 'Outside'

points['zone'] = points.apply(classify_zone, axis=1)


for zone_name, zone_color in zip(['Critical', 'Less Critical', 'Least Critical', 'Safe'], zone_colors_points):
    zone_points = points[(points['zone'] == zone_name)]
    plt.scatter(zone_points['x'], zone_points['y'], c=zone_color, s=50, 
                edgecolors='black', label=f'{zone_name} Points')


outside_points = points[points['zone'] == 'Outside']
plt.scatter(outside_points['x'], outside_points['y'], c='gray', s=50, 
            edgecolors='black', label='Outside Points')


plt.plot(0, 0, 'bo', markersize=10, label='Robot (0,0)')

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Triangle Area Colored by Zones with LaserScan Points')
plt.axis('equal')


margin = 0.1
xmin, ymin = np.min([V1, V2, V3], axis=0) - margin
xmax, ymax = np.max([V1, V2, V3], axis=0) + margin
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.grid(True)

patches = [Patch(color=c, label=l) for c, l in zip(
    zone_colors_fill,
    ['Critical Zone', 'Less Critical Zone', 'Least Critical Zone', 'Safe Zone']
)]
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=patches + handles, loc='upper right')

plt.show()
