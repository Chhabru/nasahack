# ﷽
# Uses the provided Jupyter notebook for some basic code.
# Import libraries
import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.filter import bandpass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset
from sklearn.linear_model import LinearRegression, LogisticRegression

#Load file of files
mseed_file = input("Enter the full path of an mseed file: ")
st = read(mseed_file)


# This is how you get the data and the time, which is in seconds
tr = st.traces[0].copy()
tr_times = tr.times()
tr_data = tr.data

# Plot the trace! 
fig,(ax,ax2,ax5,ax6) = plt.subplots(4,1,figsize=(14,8))
ax.plot(tr_times,tr_data)

# Make the plot pretty
ax.set_xlim([min(tr_times),max(tr_times)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax.set_title(f'{mseed_file}', fontweight='bold')

minfreq = 0.01  # Minimum frequency (Hz)
maxfreq = 0.5   # Maximum frequency (Hz)
tr_data = bandpass(tr_data, minfreq, maxfreq, df=tr.stats.sampling_rate)

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
ax2.plot(tr_times,cft,c='red')
ax2.set_xlim([min(tr_times),max(tr_times)])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Characteristic function')

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
# absdev1 = np.abs(cft-thr_off)
# ax3.plot(tr_times,absdev1,c='purple')
# ax3.set_xlim([min(tr_times),max(tr_times)])
# ax3.set_xlabel('Time (s)')
# ax3.set_ylabel('Absolute Deviation')

# secondmean = outliers(absdev1)[1].mean()
# # Plot absolute deviation again
# absdev2 = np.abs(absdev1-secondmean)
# ax4.plot(tr_times,absdev2,c='purple')
# ax4.set_xlim([min(tr_times),max(tr_times)])
# ax4.set_xlabel('Time (s)')
# ax4.set_ylabel('Absolute Deviation II')

# #Get the largest spike X-Value
# sorted_absdev2 = np.sort(absdev2)[::-1]
# largesty = sorted_absdev2[0]
# x_largy = tr_times[np.where(absdev2 == largesty)]
# print(x_largy)
# print(largesty)

on_off = np.array(trigger_onset(cft, np.percentile(cft, 80), np.percentile(cft, 60)))
# The first column contains the indices where the trigger is turned "on". 
# The second column contains the indices where the trigger is turned "off".

if on_off.ndim == 1:
    on_off = on_off.reshape(-1, 2)

sorted_tr_data = np.sort(tr_data)[::-1]
largestQuake = sorted_tr_data[0]
largestTime = tr_times[np.where(tr_data == largestQuake)]
# Find the closest 'on' time to largestTime by computing absolute differences
# This gives the index of the closest "on" event to largestTime
closest_on_index = np.argmin(np.abs(tr_times[on_off[:, 0]] - largestTime))

# This gives the actual "on" time value
closest_on_value = tr_times[on_off[closest_on_index, 0]]

# Now to access the corresponding "off" time
off_val = tr_times[on_off[closest_on_index, 1]]
# Plot seismogram
ax5.plot(tr_times,tr_data,c='orange')
ax5.set_xlim([min(tr_times),max(tr_times)])

# Plot on and off triggers
# for i in np.arange(0,len(on_off)):
#     triggers = on_off[i]
#     ax5.axvline(x = tr_times[triggers[0]], color='aqua', label='Trig. On')
#     ax5.axvline(x = tr_times[triggers[1]], color='purple', label='Trig. Off')
ax5.axvline(x = closest_on_value, c='aqua', label='Trig. On')
ax5.axvline(x = off_val, c='green', label='Trig. Off')

ax5.legend()

quakeindices = np.where((tr_times >= closest_on_value) & (tr_times <= off_val))

quakex = tr_times[quakeindices]
quakey = tr_data[quakeindices]


ax6.plot(quakex, quakey, c = 'purple')
ax6.set_xlim([min(quakex),max(quakex)])

quakex = np.array(quakex, dtype=float)
quakey = np.array(quakey, dtype=float)


# Export detected event to a catalog (adjust path as needed)
detection_time = {
    'time_rel': quakex.tolist(),
    'velocity': quakey.tolist(),
}

df_detections = pd.DataFrame(detection_time)

df_detections.to_csv('/Users/zaydmohammad/Documents/NASAHack/space_apps_2024_seismic_detection/high_confidence_detections.csv', index=False)

plt.show()
