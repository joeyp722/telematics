import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.widgets import Slider, Button, RadioButtons

import database_io as db
import parameters as par

# Load parameters for KMeans algorithm.
n_clusters = par.n_clusters
batch_size = par.batch_size
n_init = par.n_init
max_no_improvement = par.max_no_improvement

# Load parameters for KNearestNeighbors.
n_neighbors = par.n_neighbors

# Get data from database and construct dataframe.
df = db.get_content_database()

# Get values from dataframe.
columns = np.array(["Engine Coolant Temperature (deg C)", "Calculated Engine Load Value (%)", "Absolute Throttle Position (%)", "Intake Air Temperature (deg C)", "Engine RPM (rpm)", "Intake Manifold Absolute Pressure (kPa)", "Vehicle Speed (km/h)", "Long Term Fuel Trim Bank 1 (%)", "O2 Bank 1 - Sensor 1 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 1 - Short Term Fuel Trim (%)", "O2 Bank 1 - Sensor 2 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 2 - Short Term Fuel Trim (%)", "Short Term Fuel Trim Bank 1 (%)", "Timing Advance for #1 cylinder (deg )"])
X = df.loc[:,columns].values

# Compute clustering with KMeans
k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init)
k_means.fit(X)

# Get cluster centers and classification labels.
k_means_cluster_centers = k_means.cluster_centers_
y = pairwise_distances_argmin(X, k_means_cluster_centers)

# Apply labels to dataset.
neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
neigh.fit(X, y)

# Set up GUI.
#################################################################################

# Configure layout.
fig, ax = plt.subplots(figsize=(19,9))
plt.subplots_adjust(left=0.68, right = 0.98)

# Define classes.
classes = ['Class 0', 'Class 1', 'Class 2']

# Default values for sliders.
default_values = [80, 30, 30, 20, 2000, 50, 60, 3, 0.45, 99, 0.45, 3 , 0, 5]

# Get probalities from the KNearestNeighbors algorithm.
prob_array = neigh.predict_proba([default_values])

# Creating Box chart.
l = ax.bar(classes, prob_array[0])

# Set dimensions for sliders.
axcolor = 'lightgoldenrodyellow'
axcol0 = plt.axes([0.2, 0.75, 0.40, 0.03], facecolor=axcolor)
axcol1 = plt.axes([0.2, 0.70, 0.40, 0.03], facecolor=axcolor)
axcol2 = plt.axes([0.2, 0.65, 0.40, 0.03], facecolor=axcolor)
axcol3 = plt.axes([0.2, 0.60, 0.40, 0.03], facecolor=axcolor)
axcol4 = plt.axes([0.2, 0.55, 0.40, 0.03], facecolor=axcolor)
axcol5 = plt.axes([0.2, 0.50, 0.40, 0.03], facecolor=axcolor)
axcol6 = plt.axes([0.2, 0.45, 0.40, 0.03], facecolor=axcolor)
axcol7 = plt.axes([0.2, 0.40, 0.40, 0.03], facecolor=axcolor)
axcol8 = plt.axes([0.2, 0.35, 0.40, 0.03], facecolor=axcolor)
axcol9 = plt.axes([0.2, 0.30, 0.40, 0.03], facecolor=axcolor)
axcol10 = plt.axes([0.2, 0.25, 0.40, 0.03], facecolor=axcolor)
axcol11 = plt.axes([0.2, 0.20, 0.40, 0.03], facecolor=axcolor)
axcol12 = plt.axes([0.2, 0.15, 0.40, 0.03], facecolor=axcolor)
axcol13 = plt.axes([0.2, 0.10, 0.40, 0.03], facecolor=axcolor)

# Define sliders with min, max and default values.
scol0 = Slider(axcol0, columns[0], 0.1, 100, valinit = default_values[0])
scol1 = Slider(axcol1, columns[1], 0.1, 100, valinit = default_values[1])
scol2 = Slider(axcol2, columns[2], 0.1, 100, valinit = default_values[2])
scol3 = Slider(axcol3, columns[3], 0.1, 50, valinit = default_values[3])
scol4 = Slider(axcol4, columns[4], 0.1, 7000, valinit = default_values[4])
scol5 = Slider(axcol5, columns[5], 0.1, 100, valinit = default_values[5])
scol6 = Slider(axcol6, columns[6], 0.1, 200, valinit = default_values[6])
scol7 = Slider(axcol7, columns[7], -15, 15, valinit = default_values[7])
scol8 = Slider(axcol8, columns[8], 0.1, 1, valinit = default_values[8])
scol9 = Slider(axcol9, columns[9], 0.1, 100, valinit = default_values[9])
scol10 = Slider(axcol10, columns[10], 0.1, 1, valinit = default_values[10])
scol11 = Slider(axcol11, columns[11], -15, 15, valinit = default_values[11])
scol12 = Slider(axcol12, columns[12], -15, 15, valinit = default_values[12])
scol13 = Slider(axcol13, columns[13], -50, 50, valinit = default_values[13])

# Function to update values based on sliders.
def update(val):
    vcol0 = scol0.val
    vcol1 = scol1.val
    vcol2 = scol2.val
    vcol3 = scol3.val
    vcol4 = scol4.val
    vcol5 = scol5.val
    vcol6 = scol6.val
    vcol7 = scol7.val
    vcol8 = scol8.val
    vcol9 = scol9.val
    vcol10 = scol0.val
    vcol11 = scol1.val
    vcol12 = scol2.val
    vcol13 = scol3.val

    # Creating value array.
    values=[vcol0, vcol1, vcol2, vcol3, vcol4, vcol5, vcol6, vcol7, vcol8, vcol9, vcol10, vcol11, vcol12, vcol13]

    # Get probalities from the KNearestNeighbors algorithm.
    prob_array = neigh.predict_proba([values])

    # Update bar graph.
    l[0].set_height(prob_array[0,0])
    l[1].set_height(prob_array[0,1])
    l[2].set_height(prob_array[0,2])

# Call function to update values that the sliders control.
scol0.on_changed(update)
scol1.on_changed(update)
scol2.on_changed(update)
scol3.on_changed(update)
scol4.on_changed(update)
scol5.on_changed(update)
scol6.on_changed(update)
scol7.on_changed(update)
scol8.on_changed(update)
scol9.on_changed(update)
scol10.on_changed(update)
scol11.on_changed(update)
scol12.on_changed(update)

# Define reset button.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# Function to reset values.
def reset(event):
    scol0.reset()
    scol1.reset()
    scol2.reset()
    scol3.reset()
    scol4.reset()
    scol5.reset()
    scol6.reset()
    scol7.reset()
    scol8.reset()
    scol9.reset()
    scol10.reset()
    scol11.reset()
    scol12.reset()
    scol13.reset()

# Call rest function when the button is pushed.
button.on_clicked(reset)

# Show figure.
plt.show()
