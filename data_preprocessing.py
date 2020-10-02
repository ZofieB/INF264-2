import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier

#read file to input
df = pd.read_csv('data.csv', header = None)
X = df.to_numpy()
X = np.array([[int(value) for value in array_element] for array_element in X[1:]])
fra_time = X[:, 3]
volume_til_sntr = X[:, 4]
volume_til_dnp = X[:, 5]
volume_totalt = X[:, 6]

#plot datapoints per hour for both directions
fig = plt.figure(1, figsize=(10,10))
plt.ylim((0, 500))
ylabels = np.arange(10)
ylabels = ylabels * 50
fra_time_mod = fra_time + 0.4
plt.yticks(ylabels, ylabels)
plt.scatter(fra_time,volume_til_sntr, label="Volume to centre")
plt.scatter(fra_time_mod,volume_til_dnp, label="Volume to Danmarksplass")
plt.legend()
#plt.show()

#convert dates to date objects and sort into weekdays
mon = []
tue = []
wed = []
thu = []
fri = []
sat = []
sun = []
weekdays = [mon, tue, wed, thu, fri, sat, sun]
dates = []
for datapoint in X:
    datetime_point = datetime.datetime(datapoint[0], datapoint[1], datapoint[2], hour = datapoint[3])
    new_datapoint = [datetime_point, datapoint[4], datapoint[5], datapoint[6]]
    dates.append(new_datapoint)
    day_index = datetime_point.weekday()
    weekdays[day_index].append(new_datapoint)

#plot every weekday seperately
# Display scatterplots of target prices with respect to each of the 13 features:
colors = ['cornflowerblue',
          'tab:orange',
          'tab:green',
          'r',
          'tab:purple',
          'tab:brown',
          'tab:pink',
          'b',
          'tab:olive',
          'tab:cyan',
          'lightcoral',
          'chocolate',
          'springgreen',
          'g']
weekday_strings = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
for day in range(len(weekdays)):
    weekday_points_np = np.array(weekdays[day])
    datetime_arr = weekday_points_np[:, 0]
    fra_time_temp = [date.hour for date in datetime_arr]
    fra_time_mod_temp = np.array(fra_time_temp) + 0.4
    vol_til_sntr_temp = weekday_points_np[:, 1]
    vol_til_dnp_temp = weekday_points_np[:, 2]
    plt.figure(2, figsize=(16, 10))
    if day < len(weekdays) - 1:
        plot_idx = day+1
    else:
        plot_idx = 7
    plt.subplot(4, 2, plot_idx)
    plt.scatter(fra_time_temp,vol_til_sntr_temp, label="Volume to centre", c = colors[day])
    plt.scatter(fra_time_mod_temp,vol_til_dnp_temp, label="Volume to Danmarksplass", c = colors[day + 7])
    plt.xlabel(weekday_strings[day])
    plt.ylabel('Volume')
    plt.legend()
plt.suptitle("Volume in both directions for each weekday")
plt.show()


#Models

#Linear Regression with polynomial basic function
mse_train = []
mse_val = []
r2_train = []
r2_val = []

for k in range(6):
    poly_model = make_pipeline(PolynomialFeatures(k),LinearRegression())
    poly_model.fit(X_train, Y_train)

    Y_train_pred_poly = poly_model.predict(X_train)
    Y_val_pred_poly = model.predict(X_val)

    mse_train.append(mean_squared_error(Y_train, Y_train_pred_poly))
    r2_train.append(r2_score(Y_train, Y_train_pred_poly))
    mse_val.append(mean_squared_error(Y_val, Y_val_pred_poly))
    r2_val.append(r2_score(Y_val, Y_val_pred_poly))

plt.figure(figsize=(10,10))
plt.ylim(min(min(mse_train),min(mse_val)), min(max(max(mse_train), max(mse_val)), 300))
plt.plot(x_axis, mse_train, label="Training")
plt.plot(x_axis, mse_val, label="Validation")
plt.legend()

plt.figure(figsize=(10,10))
plt.ylim(0,1)
plt.plot(range(6), r2_train, label="Training")
plt.plot(range(6), r2_val, label="Validation")
plt.legend()

#Linear Regression with gaussian basic function
