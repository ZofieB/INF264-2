import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

#read file to input
df = pd.read_csv('data.csv', header = None)
X_raw = df.to_numpy()
X_raw = np.array([[int(value) for value in array_element] for array_element in X_raw[1:]])
fra_time = X_raw[:, 3]
volume_til_sntr = X_raw[:, 4]
volume_til_dnp = X_raw[:, 5]
volume_totalt = X_raw[:, 6]

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
jan = []
feb = []
mar = []
apr = []
mai = []
jun = []
jul = []
aug = []
sep = []
octo = []
nov = []
dec = []
months = [jan, feb, mar, apr, mai, jun, jul, aug, sep, octo, nov, dec]
for datapoint in X_raw:
    datetime_point = datetime.datetime(datapoint[0], datapoint[1], datapoint[2], hour = datapoint[3])
    new_datapoint = [datetime_point, datapoint[4], datapoint[5], datapoint[6]]
    dates.append(new_datapoint)
    day_indeX_raw = datetime_point.weekday()
    weekdays[day_indeX_raw].append(new_datapoint)
    months[datetime_point.month - 1].append(new_datapoint)

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
        plot_idX_raw = day+1
    else:
        plot_idX_raw = 7
    plt.subplot(4, 2, plot_idX_raw)
    plt.scatter(fra_time_temp,vol_til_sntr_temp, label="Volume to centre", c = colors[day])
    plt.scatter(fra_time_mod_temp,vol_til_dnp_temp, label="Volume to Danmarksplass", c = colors[day + 7])
    plt.xlabel(weekday_strings[day])
    plt.ylabel('Volume')
    plt.legend()
plt.suptitle("Volume in both directions for each weekday")

month_strings = ["January", "Feburary", "March", "April", "Mai", "June", "July", "August", "September", "October", "November", "December"]

for month in range(len(months)):
    month_points_np = np.array(months[month])
    datetime_arr = month_points_np[:, 0]
    fra_time_temp = [date.hour for date in datetime_arr]
    fra_time_mod_temp = np.array(fra_time_temp) + 0.4
    vol_til_sntr_temp = month_points_np[:, 1]
    vol_til_dnp_temp = month_points_np[:, 2]
    plt.figure(3, figsize=(16, 10))
    if month < len(months) - 1:
        plot_idX_raw = month + 1
    else:
        plot_idX_raw = 12
    plt.subplot(3, 4, plot_idX_raw)
    plt.scatter(fra_time_temp,vol_til_sntr_temp, label="Volume to centre", c = colors[0])
    plt.scatter(fra_time_mod_temp,vol_til_dnp_temp, label="Volume to Danmarksplass", c = colors[1])
    plt.xlabel(month_strings[month])
    plt.ylabel('Volume')
    plt.legend()
plt.suptitle("Volume in both directions for each month")
plt.show()

#creating the feature matriX_raw:
#0 weekday 1 saturday 2 sunday/free day 3 hours continuous feature
fe_datapoints = []
Y_dnp_list = []
Y_sntr_list = []
Y_totalt_list = []
for datapoint in dates:
    day_indeX_raw = datapoint[0].weekday()
    # Christmas
    if datapoint[0].month == 12 and (datapoint[0].day == 25 or datapoint[0].day == 26):
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    #New Years
    elif datapoint[0].month == 1 and datapoint[0].day == 1:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    #Constitution day, itnernational workers day
    elif datapoint[0].month == 5 and (datapoint[0].day == 17 or datapoint[0].day == 1):
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    #all the easters
    elif datapoint[0].year == 2015 and datapoint[0].month == 4 and (datapoint[0].day == 2 or datapoint[0].day == 3 or datapoint[0].day == 6):
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2016 and datapoint[0].month == 3 and (datapoint[0].day == 24 or datapoint[0].day == 25 or datapoint[0].day == 28):
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2017 and datapoint[0].month == 4 and (datapoint[0].day == 13 or datapoint[0].day == 14 or datapoint[0].day == 17):
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif (datapoint[0].year == 2018 and datapoint[0].month == 4 and datapoint[0].day == 2) or (datapoint[0].month == 2 and (datapoint[0].day == 30 or datapoint[0].day == 29)):
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2019 and datapoint[0].month == 4 and (datapoint[0].day == 18 or datapoint[0].day == 19 or datapoint[0].day == 22):
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    #Ascention days
    elif datapoint[0].year == 2015 and datapoint[0].month == 5 and datapoint[0].day == 14:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2016 and datapoint[0].month == 5 and datapoint[0].day == 5:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2017 and datapoint[0].month == 5 and datapoint[0].day == 25:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2018 and datapoint[0].month == 5 and datapoint[0].day == 10:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2019 and datapoint[0].month == 5 and datapoint[0].day == 30:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    #Whit mondays
    elif datapoint[0].year == 2015 and datapoint[0].month == 5 and datapoint[0].day == 25:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2016 and datapoint[0].month == 5 and datapoint[0].day == 16:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2017 and datapoint[0].month == 6 and datapoint[0].day == 5:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2018 and datapoint[0].month == 5 and datapoint[0].day == 21:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    elif datapoint[0].year == 2019 and datapoint[0].month == 6 and datapoint[0].day == 10:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    #saturdays
    elif day_indeX_raw == 5:
        fe_datapoints.append([0,1,0,datapoint[0].hour])
    #sundays
    elif day_indeX_raw == 6:
        fe_datapoints.append([0,0,1,datapoint[0].hour])
    #normal weekdays
    else:
        fe_datapoints.append([1,0,0,datapoint[0].hour])
    #append label values
    Y_sntr_list.append(datapoint[1])
    Y_dnp_list.append(datapoint[2])
    Y_totalt_list.append(datapoint[3])

X = np.array(fe_datapoints)
Y_sntr = np.array(Y_sntr_list)
Y_dnp = np.array(Y_dnp_list)
Y_totalt = np.array(Y_totalt_list)