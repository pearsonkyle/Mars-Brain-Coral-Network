
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as pch
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from skimage.transform import rescale

# Set figure size in inches to existing scrollwork dimensions
figsize = (20,10)
north = [0, 90]
south = [0, -90]
east = [90, 0]
west = [270, 0]

large_proj = ccrs.EckertIV()

dtm = "figures/Mars_HRSC_MOLA_BlendDEM_Global_200mp_1024.jpeg"
# https://astrogeology.usgs.gov/search/map/Mars/Topography/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2

elevation_map = plt.imread(dtm)
elevation_map = np.array(elevation_map, dtype=float)
elevation_map = rescale(elevation_map, 2, order=1) # upscale 2x

def make_figure(ch=79):
    fig = plt.figure(figsize=figsize)
    grid = matplotlib.gridspec.GridSpec(6040, 12002)
    ax = fig.add_subplot(grid[40:6000, 40:12002-40], projection=ccrs.PlateCarree())
    return fig, ax


# load all the image data => negative labels
sv = json.load(open('output/statevector.json'))

# {'url': 'https://hirise-pds.lpl.arizona.edu/download/PDS/RDR/ESP/ORB_044500_044599/ESP_044505_2265/ESP_044505_2265_RED.JP2',
#  'mask_sum': 205,
#  'lat': 45.9184591456235,
#  'lon': 41.7063466340995,
#  'scale': 0.25,
#  'title': 'Layers in crater deposit in Protonilus Mensae',
#  'local_time': 14.91269,
#  'acq_date': '2016-01-24T08:16:05.301',
#  'segmentation_mask_sum': 9873215,
#  'segmentation_percent': 0.543835481386582}

# create figure and axes for map
fig, ax = make_figure()
im = ax.imshow(elevation_map, cmap='binary_r', vmin=5,vmax=175, extent=[-180, 180, -90, 90], origin='upper', transform=ccrs.PlateCarree())

# for each in 'empty' red plot
for i, key in enumerate(sv['empty'].keys()):
    data = sv['empty'][key]
    lon = data['lon']
    lat = data['lat']
    ax.plot(lon, lat, marker='o', ls='none', color='red', markersize=1, transform=ccrs.PlateCarree(), alpha=0.5)

# plot the detections as green circles
for i, key in enumerate(sv['data'].keys()):
    data = sv['data'][key]
    lon = data['lon']
    lat = data['lat']
    mask_sum = data['mask_sum']
    scale = data['scale']
    title = data['title']
    local_time = data['local_time']
    acq_date = data['acq_date']
    #segmentation_mask_sum = data['segmentation_mask_sum']
    segmentation_percent = data.get('segmentation_percent',0)

    # ignore points at the poles
    if lat > 75 or lat < -75:
        continue

    if segmentation_percent < 0.5:
        ax.plot(lon, lat, marker='o', ls='none', color='orange', markersize=1.5, transform=ccrs.PlateCarree(), alpha=0.5)
    else:
        ax.plot(lon, lat, marker='o', ls='none', color='limegreen', markersize=2, transform=ccrs.PlateCarree(), alpha=0.75)

ax.set_xlabel('Longitude [deg]', fontsize=14)
ax.set_ylabel('Latitude [deg]', fontsize=14)
ax.set_title('Mars Brain Coral Detections', fontsize=16)
ax.set_global()

# add labels on x-axis
ax.set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())
ax.set_xticklabels(['180°W', '90°W', '0°', '90°E', '180°E'], fontsize=12)
ax.set_yticks([-90, -45, 0, 45, 90], crs=ccrs.PlateCarree())
ax.set_yticklabels(['90°S', '45°S', '0°', '45°N', '90°N'], fontsize=12)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig('mars_map_brain_coral.png')
plt.show()


# create a latitude distribution plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
latitudes = np.array([data['lat'] for data in sv['data'].values()])
mask = (latitudes > -75) & (latitudes < 75)
latitudes = latitudes[mask]
segmentation_percent = np.array([data.get('segmentation_percent',0) for data in sv['data'].values()])
segmentation_percent = segmentation_percent[mask]
mask = segmentation_percent > 0.5
big_latitudes = latitudes[mask]
ax.hist(big_latitudes, bins=np.linspace(-75,75,50), color='limegreen', alpha=0.5, label='Good')
mask = (segmentation_percent < 0.5) & (segmentation_percent > 0.05)
small_latitudes = latitudes[mask]
ax.hist(small_latitudes, bins=np.linspace(-75,75,50), color='orange', alpha=0.5, label='Maybe')
ax.legend()
ax.grid(True,ls='--',alpha=0.5)
ax.set_xlabel('Latitude [deg]', fontsize=14)
ax.set_ylabel('Counts', fontsize=14)
ax.set_title('Distribution of Brain Coral Detections', fontsize=16)
plt.tight_layout()
plt.savefig('mars_brain_coral_latitudes.png')
plt.show()
