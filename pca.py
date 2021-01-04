import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import database_io as db

# Get data from database and construct dataframe.
df = db.get_content_database()

# Get values from dataframe.
columns = np.array(["Engine Coolant Temperature (deg C)", "Calculated Engine Load Value (%)", "Absolute Throttle Position (%)", "Intake Air Temperature (deg C)", "Engine RPM (rpm)", "Intake Manifold Absolute Pressure (kPa)", "Vehicle Speed (km/h)", "Long Term Fuel Trim Bank 1 (%)", "O2 Bank 1 - Sensor 1 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 1 - Short Term Fuel Trim (%)", "O2 Bank 1 - Sensor 2 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 2 - Short Term Fuel Trim (%)", "Short Term Fuel Trim Bank 1 (%)", "Timing Advance for #1 cylinder (deg )"])
x = df.loc[:,columns].values

# The number of n_components is selected on the 95% variance or selected manually. Set up pipeline that normalizes the data and executes the pca.
# pipeline = Pipeline([('scaling', MinMaxScaler()), ('pca', PCA(n_components=0.95))])
pipeline = Pipeline([('scaling', MinMaxScaler()), ('pca', PCA(n_components=5))])
x_norm = pipeline.fit_transform(x)

# Plot results from pca algorithm.
plt.plot(x_norm)
legend_x = 1
legend_y = 0.5
legend = ["pca-1", "pca-2", "pca-3", "pca-4", "pca-5", "pca-6", "pca-7", "pca-8", "pca-9", "pca-10", "pca-11", "pca-12", "pca-13", "pca-14", "pca-15"]
plt.xlabel("Data row")
plt.ylabel("PCA")
plt.legend(legend, loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.show()
