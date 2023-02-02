# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:32:10 2022

@author: zcope
"""

from distutils.dir_util import copy_tree
import os, sys
import numpy as np

def main(qf_arrs, manual_dz):
    #Print QF input files
    dom = qf_arrs.dom
    wind_sensors = qf_arrs.wind_sensors
    global QF_PATH
    QF_PATH = dom.QF_PATH
    ws_keys = list(wind_sensors.keys())
    qf_arrs.export_fuel()   #Export QF fuel
    print_gridlist(dom)
    print_QFire_Advanced_User_Inputs_inp()
    print_QFire_Bldg_Advanced_User_Inputs_inp()
    print_QFire_Plume_Advanced_User_Inputs_inp()
    print_QP_buildout_inp()
    print_QUIC_fire_inp(dom, wind_sensors[ws_keys[0]])
    print_QU_buildings_inp()
    print_QU_fileoptions_inp()
    print_QU_metparams_inp(wind_sensors)
    print_QU_movingcoords_inp()
    print_QU_simparams_inp(dom, wind_sensors[ws_keys[0]], qf_arrs, manual_dz=manual_dz)
    print_rasterorigin_txt()
    print_Runtime_Advanced_User_Inputs_inp()
    for k in ws_keys:
        print_sensor_inp(wind_sensors[k])
    print_QU_TopoInputs_inp(flat=not qf_arrs.use_topo)
    
    src = dom.ToCopy
    dst = QF_PATH
    copy_tree(src, dst)
    print("Run setup complete")
    
def print_gridlist(dom):
    with open(os.path.join(QF_PATH,'gridlist'), 'w') as input_file:
        input_file.write("       n={} m={} l={} aa1=1.\n".format(dom.nx, dom.ny, dom.nz))
        input_file.write("       dx={} dy={} dz=1.\n".format(dom.dx, dom.dy))


def print_QFire_Advanced_User_Inputs_inp():
    with open(os.path.join(QF_PATH,'QFire_Advanced_User_Inputs.inp'), 'w') as input_file:
        input_file.write("0.05			! N/A, fraction of the cells on fire that will launch a firebrand\n")
        input_file.write("40.			! N/A, scaling factor of the radius represented by the firebrands launched\n")
        input_file.write("1				! s, time step for the firebrands trajectory calculation\n")
        input_file.write("10				! s, how often to launch firebrands\n")
        input_file.write("500			! N/A, number of firebrands distributed over the landing area\n")
        input_file.write("20.			! N/A, FB_FRACTION_LAUNCHED_to_RT_ratio\n")
        input_file.write("50.			! N/A, min_b_value_coef\n")
        input_file.write("0.75			! N/A, fb_frac_of_max_size\n")
        input_file.write("180			! s, germination_delay\n")
        input_file.write("5.				! N/A, fraction of the cell on fire (to scale w)\n")
        input_file.write("50				! N/A, minimum number of ignitions via firebrands at a point\n")
        input_file.write("100			! N/A, maximum number of ignitions via firebrands at a point\n")
        input_file.write("0.523598		! rad, min_theta_value (pi/6)\n")
        input_file.write("0.03        ! m, fb thickness\n")


def print_QFire_Bldg_Advanced_User_Inputs_inp():
    with open(os.path.join(QF_PATH,'QFire_Bldg_Advanced_User_Inputs.inp'), 'w') as input_file:
        input_file.write("1			! N/A, flag to convert QUIC-URB buildings to fuel (0 = no, 1 = yes)\n")
        input_file.write("0.5		! kg/m3, thin fuel density within buildings (if no fuel is specified)\n")
        input_file.write("2.			! N/A, attenuation coefficient within buildings\n")
        input_file.write("0.1	   ! m, surface roughness within buildings\n")
        input_file.write("1			! N/A, flag to convert fuel to canopy for winds (0 = no, 1 = yes)\n")
        input_file.write("1			! N/A, update canopy winds when fuel is consumed (0 = no, 1 = yes)\n")
        input_file.write("1			! N/A, attenuation coefficient within fuel for the Cionco profile (default = 1)\n")
        input_file.write("0.1	   ! m, surface roughness within fuel (default = 0.1 m)\n")
        input_file.write("\n")


def print_QFire_Plume_Advanced_User_Inputs_inp():
    ws = 6.0
    with open(os.path.join(QF_PATH,'QFire_Plume_Advanced_User_Inputs.inp'), 'w') as input_file:
        input_file.write("150000			! N/A, max number of plume at each time step\n")
        input_file.write("0.1			! m/s, minimum vertical velocity of a plume. If wc is below minimum, the plume is eliminated\n")
        input_file.write("100				! m/s, maximum vertical velocity of a plume (default 100 m/s)\n")
        #Per Rod convo on 10/20/2021 for how to calc MSR:
        if 0.1 * ws > 0.5:
            msr = round(0.5/ws, 3)
        else: msr = 0.1
        #Ig min speed ratio * wind speed <= 0.5
        input_file.write("{}			! N/A, minimum speed ratio (plume vertical velocity/wind speed). If below, the plume is eliminated\n".format(msr))
        input_file.write("0.					! 1/s2, brunt vaisala frequency squared\n")
        input_file.write("1					! N/A creeping flag: 0 = off, 1 = on\n")
        input_file.write("0					! Flag time step (0 = constant, 1 = adaptive)\n")
        input_file.write("1              ! s, flag = 0: plumes time step plume, flag = 1: minimum plumes time step\n")
        input_file.write("1              ! sor option (0 = reset lambda at each call, 1 = keep)\n")
        input_file.write("10             ! alpha 2 - plume centerline\n")
        input_file.write("2              ! alpha 2 - plume edges\n")
        input_file.write("30.            ! deg, max angle between plumes to merging\n")
        input_file.write("0.7            ! N/A, fraction of overlap of plumes for merging\n")
        input_file.write("1              ! N/A, which wplume-to-grid scheme: 0 = new, 1 = old\n")
        input_file.write("10             ! N/A, number of points per edge for new scheme (only new scheme)\n")
        input_file.write("1              ! N/A, w computation: 1 = max(w), 0 = sum(w^3)\n")


def print_QP_buildout_inp():
    with open(os.path.join(QF_PATH,'QP_buildout.inp'), 'w') as input_file:
        input_file.write("           0  ! total number of buildings\n")
        input_file.write("           0  ! total number of vegitative canopies\n")


def print_QUIC_fire_inp(dom, wind_sensor):
    energy_packets = 100 # change based on ignition method?
    with open(os.path.join(QF_PATH,'QUIC_fire.inp'), 'w') as input_file:
        input_file.write("1					! Fire flag: 1 = run fire; 0 = no fire\n")
        input_file.write("-1				! Random number generator: -1: use time and date, any other integer > 0 is used as the seed\n")
        input_file.write("! FIRE TIMES\n")
        input_file.write("{}		! When the fire is ignited in Unix Epoch time (integer seconds since 1970/1/1 00:00:00). Must be greater or equal to the time of the first wind\n".format(wind_sensor.times[0]))
        input_file.write("{}				! Total simulation time for the fire [s]\n".format(dom.sim_time))
        input_file.write("1					! time step for the fire simulation [s]\n")
        input_file.write("1					! Number of fire time steps done before updating the quic wind field (integer, >= 1)\n")
        input_file.write("100					! After how many fire time steps to print out fire-related files (excluding emissions and radiation)\n")
        input_file.write("100					! After how many quic updates to print out wind-related files\n")
        input_file.write("4					! After how many fire time steps to average emissions and radiation\n")
        input_file.write("2					! After how many quic updates to print out averaged wind-related files\n")
        input_file.write("! FIRE GRID\n")
        input_file.write("{}					! Number of vertical layers of fire grid cells (integer)\n".format(dom.nz))
        input_file.write("0					! Vertical stretching flag: 0 = uniform dz, 1 = custom\n")
        input_file.write("1.             ! m, dz\n")
        input_file.write("! FILE PATH\n")
        input_file.write("\"\"\n")
        input_file.write("1              ! Fuel types are in separate files\n")
        input_file.write("1              ! File is stream (1) or headers (2)\n")
        input_file.write("! FUEL\n")
        input_file.write("4					! fuel density flag: 1 = uniform; 2 = provided thru QF_FuelDensity.inp, 3 = Firetech files for quic grid, 4 = Firetech files for different grid (need interpolation)\n")
        input_file.write("4					! fuel moisture flag: 1 = uniform; 2 = provided thru QF_FuelMoisture.inp, 3 = Firetech files for quic grid, 4 = Firetech files for different grid (need interpolation)\n")
        #Might want to change back
        # input_file.write("5					! fuel density flag: 1 = uniform; 2 = provided thru QF_FuelDensity.inp, 3 = Firetech files for quic grid, 4 = Firetech files for different grid (need interpolation)\n")
        # input_file.write("5					! fuel moisture flag: 1 = uniform; 2 = provided thru QF_FuelMoisture.inp, 3 = Firetech files for quic grid, 4 = Firetech files for different grid (need interpolation)\n")
        input_file.write("! IGNITION LOCATIONS\n")
        input_file.write("7					! 1 = rectangle, 2 = square ring, 3 = circular ring, 4 = file (QF_Ignitions.inp), 5 = time-dependent ignitions (QF_IgnitionPattern.inp), 6 = ignite.dat (firetech)\n")
        input_file.write("{}\n".format(energy_packets))
        input_file.write("! FIREBRANDS\n")
        input_file.write("0				! 0 = off, 1 = on\n")
        input_file.write("! OUTPUT FILES (formats depend on the grid type flag)\n")
        input_file.write("0					! Output gridded energy-to-atmosphere (3D fire grid + extra layers)\n")
        input_file.write("0					! Output compressed array reaction rate (fire grid)\n")
        input_file.write("1					! Output compressed array fuel density (fire grid)\n")
        input_file.write("0					! Output gridded wind (u,v,w,sigma) (3D fire grid)\n")
        input_file.write("0					! Output gridded QU winds with fire effects, instantaneous (QUIC-URB grid)\n")
        input_file.write("0					! Output gridded QU winds with fire effects, averaged (QUIC-URB grid)\n")
        input_file.write("0					! Output plume trajectories (ONLY FOR DEBUG)\n")
        input_file.write("0					! Output compressed array fuel moisture (fire grid)\n")
        input_file.write("0					! Output vertically-integrated % mass burnt (fire grid)\n")
        input_file.write("0					! Output trajectories firebrands\n")
        input_file.write("3					! Output compressed array emissions (fire grid)\n")
        input_file.write("0					! Output gridded thermal radiation (fire grid)\n")
        input_file.write("1              ! Output surface fire intensity at every fire time step\n")
        input_file.write("! AUTOKILL\n")
        input_file.write("0              ! Kill if the fire is out and there are no more ignitions or firebrands (0 = no, 1 = yes)\n")


def print_QU_buildings_inp():
    with open(os.path.join(QF_PATH,'QU_buildings.inp'), 'w') as input_file:
        input_file.write("!QUIC 6.26\n")
        input_file.write("0.1			!Wall roughness length (m)\n")
        input_file.write("0			!Number of Buildings\n")
        input_file.write("0			!Number of Polygon Building Nodes\n")


def print_QU_fileoptions_inp():
    with open(os.path.join(QF_PATH,'QU_fileoptions.inp'), 'w') as input_file:
        input_file.write("\n")
        input_file.write("2   !output data file format flag (1=ascii, 2=binary, 3=both)\n")
        input_file.write("0   !flag to write out non-mass conserved initial field (uofield.dat) (1=write,0=no write)\n")
        input_file.write("0   !flag to write out the file uosensorfield.dat, the initial sensor velocity field (1=write,0=no write)\n")
        input_file.write("0   !flag to write out the file QU_staggered_velocity.bin used by QUIC-Pressure(1=write,0=no write)\n")
        input_file.write("0   !flag to generate startup files\n")

def print_QU_metparams_inp(wind_sensors):
    with open(os.path.join(QF_PATH,'QU_metparams.inp'), 'w') as input_file:
        input_file.write("!QUIC 6.26\n")
        input_file.write("0 !Met input flag (0=QUIC,1=WRF,2=ITT MM5,3=HOTMAC)\n")
        input_file.write("{} !Number of measuring sites\n".format(len(wind_sensors.keys())))
        input_file.write("1 !Maximum size of data points profiles\n")
        for k in wind_sensors.keys():
            sensor = wind_sensors[k]
            input_file.write("{} !Site Name\n".format(sensor.SENSOR_NAME))
            input_file.write("!File name\n")
            input_file.write("{}.inp\n".format(sensor.SENSOR_NAME))


def print_QU_movingcoords_inp():
    with open(os.path.join(QF_PATH,'QU_movingcoords.inp'), 'w') as input_file:
        input_file.write("!QUIC 6.3\n")
        input_file.write("0   !Moving coordinates flag (0=no, 1=yes)\n")
        input_file.write("0   !Reference bearing of the ship relative to the non-rotated domain (degrees)\n")
        input_file.write("!Time (Unix Time), Ship Speed (m/s), Ship Bearing (deg), Ocean Current Speed (m/s), Ocean Current Direction (deg)\n")
        input_file.write("1488794400	0	0	0	0\n")
        input_file.write("1488794450	0	0	0	0\n")
        input_file.write("1488794500	0	0	0	0\n")
        input_file.write("1488794550	0	0	0	0\n")
        input_file.write("1488794600	0	0	0	0\n")
        input_file.write("1488794650	0	0	0	0\n")
        input_file.write("1488794700	0	0	0	0\n")
        input_file.write("1488794750	0	0	0	0\n")
        input_file.write("1488794800	0	0	0	0\n")
        input_file.write("1488794850	0	0	0	0\n")
        input_file.write("1488794900	0	0	0	0\n")
        input_file.write("1488794950	0	0	0	0\n")
        input_file.write("1488795000	0	0	0	0\n")


def print_QU_simparams_inp(dom, wind_sensor, qf_arrs, manual_dz=False):
    with open(os.path.join(QF_PATH,'QU_simparams.inp'), 'w') as input_file:
        input_file.write("!QUIC 6.26\n")
        input_file.write("{} !nx - Domain Length(X) Grid Cells\n".format(dom.nx))
        input_file.write("{} !ny - Domain Width(Y) Grid Cells\n".format(dom.ny))
        input_file.write("22 !nz - Domain Height(Z) Grid Cells\n")
        input_file.write("{} !dx (meters)\n".format(dom.dx))
        input_file.write("{} !dy (meters)\n".format(dom.dy))
        input_file.write("3 !Vertical stretching flag(0=uniform,1=custom,2=parabolic Z,3=parabolic DZ,4=exponential)\n")
        input_file.write("1.000000 !Surface dz (meters)\n")
        input_file.write("5 !Number of uniform surface cells\n")
        input_file.write("!dz array (meters)\n")
        if manual_dz:
            for z_temp in qf_arrs.dz_array:
                input_file.write("{}\n".format(z_temp))
        else:
            fuel_height = 0
            MIN_HEIGHT = 150
            for k in range(len(qf_arrs.rhof)):
                if np.max(qf_arrs.rhof[k]) != 0:
                    fuel_height = k+1
            relief = 0
            if qf_arrs.use_topo:
                relief = qf_arrs.topo.max() - qf_arrs.topo.min()
            if (relief * 3) > MIN_HEIGHT:
                height = fuel_height + (relief * 3)
            else:
                height = fuel_height + relief + MIN_HEIGHT  
            dz_array = build_parabolic_dz_array(nz=22, Lz=height, n_surf=5, dz_surf=1)
            for z_temp in dz_array:
                input_file.write("{}\n".format(z_temp))
        input_file.write("{} !total time increments\n".format(len(wind_sensor.times)))
        input_file.write("0 !UTC conversion\n")
        input_file.write("!Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n")
        for time in wind_sensor.times:
            input_file.write("{}\n".format(time))
        input_file.write("2 ! Rooftop flag (0 = none, 1 = log profile, 2 = vortex)\n")
        input_file.write("3 ! Upwind cavity flag (0 = none, 1 = Rockle, 2 = MVP, 3 = HMVP)\n")
        input_file.write("4 ! Street canyon flag (0 = none, 1 = Roeckle, 2 = CPB, 3 = exp. param. PKK, 4 = Roeckle w/ Fackrel)\n")
        input_file.write("1 ! Street intersection flag (0 = off, 1 = on)\n")
        input_file.write("4 ! Wake flag (0 = none, 1 = Rockle, 2 = modified Rockle, 3 = area scaled)\n")
        input_file.write("1 ! Sidewall flag (0 = off, 1 = on)\n")
        input_file.write("1 ! Individual tree wake flag\n")
        input_file.write("2 ! Canopy flag (0 = off, 1 = Cionco w/o wakes, 2 = Cionco w/ wakes)\n")
        input_file.write("1 ! Season flag (1 = Summer, 2 = Winter, 3 = Transition)\n")
        input_file.write("10 ! Maximum number of iterations\n")
        input_file.write("3 ! Residual reduction (Orders of Magnitude)\n")
        input_file.write("0 ! Use diffusion algorithm (0 = off, 1 = on)\n")
        input_file.write("20 ! Number of diffusion iterations\n")
        input_file.write("0 ! Domain rotation relative to true north [degrees] (cw = +)\n")
        input_file.write("0.0 ! UTMX of domain origin [m]\n")
        input_file.write("0.0 ! UTMY of domain origin [m]\n")
        input_file.write("1 ! UTM zone\n")
        input_file.write("17 ! UTM zone leter (1 = A, 2 = B, etc.)\n")
        input_file.write("0 ! QUIC-CFD Flag\n")
        input_file.write("0 ! Explosive building damage flag (1 = on)\n")
        input_file.write("0 ! Building Array Flag (1 = on)\n")

def print_QU_TopoInputs_inp(flat):
    if flat:
        topo_flag = 0
        topo_fname = '""'
    else:
        topo_flag = 11
        topo_fname = "ftelevation.dat"
    with open(os.path.join(QF_PATH, 'QU_TopoInputs.inp'), 'w') as input_file:
        input_file.write("Specify file name for custom topo (full path)\n")
        input_file.write("{}\n".format(topo_fname))
        input_file.write("{}     ! N/A, topo flag: 0 = flat, 1 = Gaussian hill, 2 = hill pass, 3 = slope mesa, 4 = canyon, 5 = custom, 6 = half circle, 7 = sinusoid, 8 = cos hill, 9 = QP_elevation.inp, 10 = terrainOutput.txt (ARA), 11 = terrain.dat (firetec)\n".format(topo_flag))
        input_file.write("0     ! N/A, smoothing method (0 = none (default for idealized terrain), 1 = blur, 2 = David's)\n")
        input_file.write("0     ! N/A, number of smoothing passes\n")
        input_file.write("100   ! N/A, number of initial SOR iterations (only if fire is run)\n")
        input_file.write("4     ! N/A, sor cycles\n")
        input_file.write("1.3   ! N/A, sor relaxation parameter (default for flat is 1.78, ignored if there is no terrain)\n")
        input_file.write("0     ! N/A, add slope flow correction (0 = no, 1 = yes)\n")
        input_file.write("Command to run slope flow code (system specific)\n")
        input_file.write("\"\"\n")


def print_rasterorigin_txt():
    with open(os.path.join(QF_PATH,'rasterorigin.txt'), 'w') as input_file:
        input_file.write("0.\n")
        input_file.write("0.\n")
        input_file.write("752265.868913356\n")
        input_file.write("3752846.04249607\n")
        input_file.write("742265.868913356\n")
        input_file.write("3742846.04249607\n")
        input_file.write("10000\n")


def print_Runtime_Advanced_User_Inputs_inp():
    with open(os.path.join(QF_PATH,'Runtime_Advanced_User_Inputs.inp'), 'w') as input_file:
        input_file.write("8  ! max number of cpu\n")


def print_sensor_inp(sensor):
    with open(os.path.join(QF_PATH,'{}.inp'.format(sensor.SENSOR_NAME)), 'w') as input_file:
        input_file.write("{} !Site Name\n".format(sensor.SENSOR_NAME))
        input_file.write("0 !Upper level flag (1 = use this profile for upper level winds)\n")
        input_file.write("50 !Upper level height (meters)\n")
        input_file.write("1 !Site Coordinate Flag (1=QUIC, 2=UTM, 3=Lat/Lon)\n")
        input_file.write("{} !X coordinate (meters)\n".format(sensor.SENSOR_X))
        input_file.write("{} !Y coordinate (meters)\n".format(sensor.SENSOR_Y))
        for i,time in enumerate(sensor.times):
            input_file.write("{} !Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n".format(time))
            input_file.write("1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n")
            input_file.write("0.01 !site zo\n")
            input_file.write("0.\n")
            input_file.write("!Height (m),Speed	(m/s), Direction (deg relative to true N)\n")
            input_file.write("{} {}	{}\n".format(sensor.SENSOR_HEIGHT, sensor.speeds[i], sensor.dirs[i]))



#Finish building
def build_parabolic_dz_array(nz=22, Lz=350, n_surf=5, dz_surf=1):
    dz_high = Lz - dz_surf * n_surf
    dz_low = 0
    z_temp = nz * dz_surf
    dz = np.zeros(nz)
    while abs(1-(z_temp/Lz)) > 0.001:
        dz_max = 1/2 * (dz_low + dz_high)
        c1 = (dz_max - dz_surf)/(nz-n_surf) ** 2
        c2 = -2 * c1 * n_surf
        c3 = dz_surf + c1 * n_surf ** 2

        dz[0:n_surf] = dz_surf

        for k in range(n_surf, nz):
            dz[k] = round((c1 * k ** 2) + (c2 * k) + c3, 2)

        z_temp = sum(dz)

        if z_temp > Lz:
            dz_high = dz_max
        elif z_temp < Lz:
            dz_low = dz_max
        else:
            break

    return dz
            
        