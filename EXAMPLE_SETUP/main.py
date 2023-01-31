# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:11:26 2022

@author: zcope
"""
##################################################################################################
#Example main.py file
#Before running install ttrs_quicfire. Instructions: https://github.com/QUIC-Fire-TT/ttrs_quicfire
#Documentation: https://github.com/QUIC-Fire-TT/ttrs_quicfire/wiki
#Script requires two folders: Shapefiles and FilesToCopy

##Shapefiles: only needs to include a ploygon of the burnplot burn_plot.shp
###Other functions may not work if the appropriate shapefile isn't being tracked

##FilesToCopy: should include all files that you want to add to your run
###You will need to incude QU_landuse.inp (a binary file that is not built with ttrs_quicfire)
###Other files are optional including: the QF executable you'd like to use and drawfire.py/a visualization script
##################################################################################################

#Standard Libraries
import os
import sys
import numpy as np
import ttrs_quicfire.buildfire as bf

OG_PATH = os.getcwd()

###Building and Tracking Shapefiles
#Track path to shapefiles that exist:
shape_paths = bf.Shapefile_Paths()

###Build domain class from shape
dom = bf.dom_from_burn_plot(shape_paths, buffer=30, QF_PATH= os.path.join(OG_PATH, 'Run'))

###Build FF Fuel domain
qf_arrs = bf.build_ff_domain(dom, FF_request=False)
#Fuel breaks
qf_arrs.build_fuelbreak() #default build 6m fuel break around the plot
qf_arrs.build_fuelbreak(shape_paths.streams, buffer = 1)
qf_arrs.build_fuelbreak(shape_paths.roads, buffer = 5)
#Update FMC
qf_arrs.update_surface_moisture(moist_in_plot=0.07, moist_out_plot=1)
#Modify Wetlands
qf_arrs.mod_wetlands(fmc=0.5, bulk_density=3)

###Build ignition file and windfield
#Wind Option 1: Manual
speeds = [0.1, 1, 2, 3, 2, 0.1, 1, 3, 4, 2, 1, 2]
directions = [90, 100, 95, 90, 80, 70, 75, 80, 85, 90, 95, 100]
times = list(range(0,(300*12),300))
avg_wind_speed = 1.5
avg_wind_dir = np.average(directions)
qf_arrs.custom_windfield(speeds=speeds, dirs=directions, times=times)

#Wind Option 2: Build windfield
# avg_wind_speed = 1.5
# avg_wind_dir = 215
#qf_arrs.calc_normal_windfield(start_speed = avg_wind_speed, start_dir = avg_wind_dir)

#Build Ignition
bf.atv_ignition(dom, wind_dir=avg_wind_dir, line_space_chain = 2)

#Build QF simulation
bf.build_qf_run(qf_arrs)