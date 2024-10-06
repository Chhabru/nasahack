# ﷽
# Import libraries
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, LogisticRegression

def largest3(arra, tr_times, tr_data):
    # Ensure 'arra' is a 2D array
    if arra.ndim == 1:
        arra = arra.reshape(-1, 2)

    # Ensure the indices are integers for accessing elements
    on_trigger_indices = arra[:, 0].astype(int)
    
    # Get the corresponding "on" times from tr_times using the indices
    on_times = tr_times[on_trigger_indices]

    # Get the corresponding 'y' values (velocity) from tr_data at those 'on' times
    y_values = tr_data[on_trigger_indices]

    # Sort the indices of y_values in descending order and get the top 3
    top_3_indices = np.argsort(y_values)[-3:][::-1]  # Fixed: Get indices of top 3
    top_3_on_times = on_times[top_3_indices]

    print("Top 3 On Times (indices):", top_3_on_times)

    top_3_off_times = []

    # Loop through the top 3 'on' times to find corresponding 'off' times
    for on_time in top_3_on_times:
        rounded_on_time = np.round(on_time, decimals=6)

        # Find the index where the 'on' time matches in the on_off array
        index = np.where(np.round(arra[:, 0], decimals=6) == rounded_on_time)[0]
        
        # Get the corresponding 'off' time if the index is valid
        if index.size > 0:
            off_time = arra[index[0], 1]
            top_3_off_times.append(off_time)
        else:
            print(f"No matching 'off' time found for on_time: {rounded_on_time}")

    # Handle the case where less than 3 off times are found
    if len(top_3_off_times) < 3:
        print("Less than 3 matching 'off' times found.")

    # Construct the result array for maximum 'off' and 'on' times
    maxlist = []
    for i in range(len(top_3_off_times)):
        maxlist.append([top_3_off_times[i], top_3_on_times[i]])

    maximum = np.a    
    print("Maximum 'off' and 'on' times:", maximum)
    return maximum

# def largest3(arra, tr_times, tr_data): 
#     # Ensure the indices are integers for accessing elements
#     on_trigger_indices = arra[:, 0].astype(int)
    
#     # Get the corresponding "on" times from tr_times using the indices
#     on_times = tr_times[on_trigger_indices]

#     # Get the corresponding 'y' values (velocity) from tr_data at those 'on' times
#     y_values = tr_data[on_trigger_indices]

#     # Sort the indices of y_values in descending order and get the top 3
#     top_3_indices = np.sort(y_values)[-3:][::-1]
#     top_3_on_times = on_times[top_3_indices]

#     print("Top 3 On Times (indices):", top_3_on_times)  # Debugging output

#     top_3_off_times = []

#     # Loop through the top 3 'on' times to find corresponding 'off' times
#     for on_time in top_3_on_times:
#         # Round on_time for comparison to handle floating point precision
#         rounded_on_time = np.round(on_time, decimals=6)

#         # Find the index where the 'on' time matches in the on_off array
#         index = np.where(np.round(arra[:, 0], decimals=6) == rounded_on_time)[0]
        
#         # Get the corresponding 'off' time if the index is valid
#         if index.size > 0:
#             off_time = arra[index[0], 1]
#             top_3_off_times.append(off_time)
#         else:
#             print(f"No matching 'off' time found for on_time: {rounded_on_time}")  # Debugging output

#     # If less than 3 off times are found, handle this scenario
#     if len(top_3_off_times) < 3:
#         print("Less than 3 matching 'off' times found.")

#     # Construct the result array for maximum 'off' and 'on' times
#     maxlist = []
#     for i in range(len(top_3_off_times)):
#         maxlist.append([top_3_off_times[i], top_3_on_times[i]])

#     maximum = np.array(maxlist)
    
#     print("Maximum 'off' and 'on' times:", maximum)  # Debugging output
#     return maximum

# def largest3(arra, tr_times, tr_data):
#     on_trigger_indices = arra[:, 0].astype(int)
    
#     # Get the corresponding "on" times from tr_times
#     on_times = tr_times[on_trigger_indices]

#     # Find the corresponding 'y' values (velocity) from tr_data at those 'on' times
#     y_values = tr_data[on_trigger_indices]

#     # Sort the indices of y_values in descending order and get the top 3
#     top_3_indices = np.argsort(y_values)[-3:][::-1]  # Get indices of the largest 3 values in descending order
#     top_3_values = y_values[top_3_indices]
#     top_3_on_times = on_times[top_3_indices]

#     print("Top 3 On Times:", top_3_on_times)  # Debugging output

#     top_3_off_times = []

#     # Loop through the top 3 'on' times
#     for on_time in top_3_on_times:
#         # Find the index where the 'on' time matches in the on_off array
#         index = np.where(arra[:, 0] == on_time)[0]
        
#         # Get the corresponding 'off' time
#         if index.size > 0:  # Ensure the index is valid
#             off_time = arra[index[0], 1]
#             top_3_off_times.append(off_time)
#         else:
#             print(f"No matching 'off' time found for on_time: {on_time}")  # Debugging output

#     maxlist = []
#     for i in range(len(top_3_off_times)):
#         maxlist.append([top_3_off_times[i], top_3_on_times[i]])

#     maximum = np.array(maxlist)

#     print("Maximum:", maximum)  # Debugging output
#     return maximum


 

#Load file of files
cat_directory = '/Users/zaydmohammad/Documents/NASAHack/space_apps_2024_seismic_detection/data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
cat = pd.read_csv(cat_file)

#Get a row
row = cat.iloc[30]
# If we want the value of relative time, we don't need to use datetime
arrival_time_rel = row['time_rel(sec)']

#Get filename
test_filename = row.filename

#Get the file from the file database
data_directory = '/Users/zaydmohammad/Documents/NASAHack/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'
mseed_file = f'{data_directory}{test_filename}.mseed'
st = read(mseed_file)


# This is how you get the data and the time, which is in seconds
tr = st.traces[0].copy()
tr_times = tr.times()
tr_data = tr.data

# Plot the trace! 
fig,ax = plt.subplots(1,1,figsize=(10,3))
ax.plot(tr_times,tr_data)

# Make the plot pretty
ax.set_xlim([min(tr_times),max(tr_times)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax.set_title(f'{test_filename}', fontweight='bold')

# Plot where the arrival time is
arrival_line = ax.axvline(x=arrival_time_rel, c='red', label='Rel. Arrival')
ax.legend(handles=[arrival_line])

# from obspy.signal.invsim import cosine_taper
# from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset

# Sampling frequency of our trace
df = tr.stats.sampling_rate

# How long should the short-term and long-term window be, in seconds?
sta_len = 120
lta_len = 600

# Run Obspy's STA/LTA to obtain a characteristic function
# This function basically calculates the ratio of amplitude between the short-term 
# and long-term windows, moving consecutively in time across the data
cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))

# Plot characteristic function
fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(tr_times,cft)
ax.set_xlim([min(tr_times),max(tr_times)])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Characteristic function')

def outliers(cft):
    # finding the 1st quartile
    q1 = np.quantile(cft, 0.25)
    
    # finding the 3rd quartile
    q3 = np.quantile(cft, 0.75)
    med = np.median(cft)
    
    # finding the iqr region
    iqr = q3-q1
    
    # finding upper and lower whiskers
    upper_bound = q3+(3*iqr)
    lower_bound = q1-(3*iqr)

    outliers = cft[(cft >= upper_bound) | (cft <= lower_bound)]
    nonoutliers = cft[(cft < upper_bound) & (cft > lower_bound)]

    return [outliers, nonoutliers]



# Play around with the on and off triggers, based on values in the characteristic function
thr_on = outliers(cft)[0].mean()
thr_off = outliers(cft)[1].mean()

# Plot absolute deviation
fig,ax = plt.subplots(1,1,figsize=(12,3))
absdev1 = np.abs(cft-thr_off)
ax.plot(tr_times,absdev1)
ax.set_xlim([min(tr_times),max(tr_times)])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Absolute Deviation')

secondmean = outliers(absdev1)[1].mean()
# Plot absolute deviation again
fig,ax = plt.subplots(1,1,figsize=(12,3))
absdev2 = np.abs(absdev1-secondmean)
ax.plot(tr_times,absdev2)
ax.set_xlim([min(tr_times),max(tr_times)])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Absolute Deviation II')

#Get the largest spike X-Value
sorted_absdev2 = np.sort(absdev2)[::-1]
largesty = sorted_absdev2[0]
x_largy = tr_times[np.where(absdev2 == largesty)]
print(x_largy)
print(largesty)

on_off = np.array(trigger_onset(cft, thr_on, thr_off))
# The first column contains the indices where the trigger is turned "on". 
# The second column contains the indices where the trigger is turned "off".

sorted_tr_data = np.sort(tr_data)[::-1]
largestQuake = sorted_tr_data[0]
largestTime = tr_times[np.where(tr_data == largestQuake)]


# Plot on and off triggers
fig,ax = plt.subplots(1,1,figsize=(12,3))
for i in np.arange(0,len(on_off)):
    triggers = on_off[i]
    ax.axvline(x = tr_times[triggers[0]], color='aqua', label='Trig. On')
    ax.axvline(x = tr_times[triggers[1]], color='purple', label='Trig. Off')
ax.axvline(x = largestTime, c='green', label='Trig. On')
ax.axvline(x = off_val, c='red', label='Trig. Off')
# Plot seismogram
ax.plot(tr_times,tr_data)
ax.set_xlim([min(tr_times),max(tr_times)])

# print(largest3(on_off, tr_times, tr_data))

plt.show()
