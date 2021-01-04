from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

import database_io as db

# Get data from database and construct dataframe.
df = db.get_content_database()

# Get values from dataframe.
columns = np.array(["Engine Coolant Temperature (deg C)", "Calculated Engine Load Value (%)", "Absolute Throttle Position (%)", "Intake Air Temperature (deg C)", "Engine RPM (rpm)", "Intake Manifold Absolute Pressure (kPa)", "Vehicle Speed (km/h)", "Long Term Fuel Trim Bank 1 (%)", "O2 Bank 1 - Sensor 1 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 1 - Short Term Fuel Trim (%)", "O2 Bank 1 - Sensor 2 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 2 - Short Term Fuel Trim (%)", "Short Term Fuel Trim Bank 1 (%)", "Timing Advance for #1 cylinder (deg )"])
X = df.loc[:,columns].values

# Show values of dataframe.
print("Original data.")
print(X.shape)
print(X)

# Apply isomap embedding.
embedding = Isomap(n_components=3)
X_transformed = embedding.fit_transform(X)

# Show result of isomap embedding.
print("Embedded data.")
print(X_transformed.shape)
print(X_transformed)

# Plot result of isomap embedding.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2], c='b', marker='o')

ax.set_xlabel('Pca1')
ax.set_ylabel('Pca2')
ax.set_zlabel('Pca3')
plt.show()
