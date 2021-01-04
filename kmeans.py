import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

import database_io as db
import parameters as par

# Choices for what the x-axis and y-axis in the plot represent.
index_x = 4
index_y = 6

# Load parameters for KMeans algorithm.
n_clusters = par.n_clusters
batch_size = par.batch_size
n_init = par.n_init
max_no_improvement = par.max_no_improvement

# Get data from database and construct dataframe.
df = db.get_content_database()

# Get values from dataframe.
columns = np.array(["Engine Coolant Temperature (deg C)", "Calculated Engine Load Value (%)", "Absolute Throttle Position (%)", "Intake Air Temperature (deg C)", "Engine RPM (rpm)", "Intake Manifold Absolute Pressure (kPa)", "Vehicle Speed (km/h)", "Long Term Fuel Trim Bank 1 (%)", "O2 Bank 1 - Sensor 1 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 1 - Short Term Fuel Trim (%)", "O2 Bank 1 - Sensor 2 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 2 - Short Term Fuel Trim (%)", "Short Term Fuel Trim Bank 1 (%)", "Timing Advance for #1 cylinder (deg )"])
X = df.loc[:,columns].values

# Compute clustering with KMeans
k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init)
k_means.fit(X)

# Compute clustering with MiniBatchKMeans
mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                      n_init=n_init, max_no_improvement=max_no_improvement, verbose=0)
mbk.fit(X)

# Make sure both algorithms have the same coloring for their clusters.
k_means_cluster_centers = k_means.cluster_centers_
order = pairwise_distances_argmin(k_means.cluster_centers_,
                                  mbk.cluster_centers_)
mbk_means_cluster_centers = mbk.cluster_centers_[order]

# Labeling according to algorithms.
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

# Set arrays for code below.
cluster_center0 = k_means_cluster_centers[0]
cluster_center1 = k_means_cluster_centers[1]
cluster_center2 = k_means_cluster_centers[2]

# Values for each cluster chosen by the KMeans algorithm.
for i in range(0, len(columns)):
    print(columns[i], cluster_center0[i], cluster_center1[i], cluster_center2[i])

# Plot result
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.20, top=0.9, wspace = 0.5)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, index_x], X[my_members, index_y], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[index_x], cluster_center[index_y], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans')

ax.set_xlabel(columns[index_x])
ax.set_ylabel(columns[index_y])

# MiniBatchKMeans
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == k
    cluster_center = mbk_means_cluster_centers[k]
    ax.plot(X[my_members, index_x], X[my_members, index_y], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('MiniBatchKMeans')

ax.set_xlabel(columns[index_x])
ax.set_ylabel(columns[index_y])

# Initialise the different array to all False
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1, 3, 3)

for k in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_means_labels == k))

identic = np.logical_not(different)
ax.plot(X[identic, index_x], X[identic, index_y], 'w',
        markerfacecolor='#bbbbbb', marker='.')
ax.plot(X[different, 0], X[different, 1], 'w',
        markerfacecolor='m', marker='.')
ax.set_title('Difference')

ax.set_xlabel(columns[index_x])
ax.set_ylabel(columns[index_y])

plt.show()
