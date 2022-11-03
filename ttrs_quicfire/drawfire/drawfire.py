# -*- coding: utf-8 -*-
# Version date: March 13, 2019
# @author: Sara Brambilla

###############################################################################
##Modified by Zachary Cope 10/13/2022
#Need To:
    # Add TTRS drawfireplus functions:
        #Power
        #ROS and BA
        #Fire line intensity
    # Make draw functions load one at a time
    # Add Arrow to fuel plots
    # Seperate plot commands
###############################################################################

#Import python packages
import math
import numpy as np
import pylab
import shutil
import os
import sys
import copy
from pyevtk.hl import gridToVTK
#Import application scripts
from .class_def import *
from .read_inputs import *
from .misc import *
from .gen_images import *


def compress_fuel_dens(qf: GridClass, flags: FlagsClass, output_folder: str):
    if flags.fuel_density == 1:
        temp_ntimes = qf.ntimes
        temp_time = qf.time
    else:
        temp_ntimes = 1
        temp_time = [qf.time[0]]

    fuel_dens = read_fireca_field_NEW("fuels-dens-00000.bin", qf, 0, output_folder)

    fuel_dens_compressed = np.sum(fuel_dens, axis=2)
    fuel_idx = np.where(fuel_dens_compressed > 0)
    no_fuel_idx = np.where(fuel_dens_compressed == 0)

    return fuel_idx, no_fuel_idx


def plot_outputs(df_classes: AllDrawFireClasses):
    #unpack df_classes
    qu, qf, ignitions, flags, fb, prj_folder, output_folder, gen_vtk, gen_gif, 
    img_specs, fuel_idx, no_fuel_idx = df_classes.class_list  
    
    #plot terrain
    plot_terrain(df_classes)

    if flags.isfire == 1:
        # print("\t-fuel density field")
        # fuel_idx, no_fuel_idx = compress_fuel_dens(qf, flags, output_folder)

        # ------- Firetech ignitions
        print("\t-initial ignitions")
        plot_ignitions(df_classes)

        # ------- fuel height (ground level only)
        print("\t-ground level fuel height")
        plot_fuelheight(df_classes)

        # ------- mass burnt (vertically-integrated)
        if flags.perc_mass_burnt == 1:
            print("\t-% mass burnt")
            plot_percmassburnt(df_classes)

        # ------- Fuel mass
        if flags.fuel_density == 1:
            #WORKING
            plot_fueldens(df_classes)
            

        plane = 1
        # ------- Emissions
        if flags.emissions == 2 or flags.emissions == 3:
            print("\t-pm emissions")
            emiss = read_fireca_field("pm_emissions-", qf.ntimes_ave, qf.time_ave, qf, 0, output_folder)
            plot_2d_field(True, qf, plane, 'xy', emiss, "Soot (log10) [g]", "pm_emissions",
                          [], img_specs, no_fuel_idx, flags)
            del emiss

        if flags.emissions == 1 or flags.emissions == 3:
            print("\t-co emissions")
            emiss = read_fireca_field("co_emissions-", qf.ntimes_ave, qf.time_ave, qf, 0, output_folder)
            plot_2d_field(True, qf, plane, 'xy', emiss, "CO (log10) [g]", "co_emissions",
                          [], img_specs, no_fuel_idx, flags)
            del emiss

        # ------- Radiation
        if flags.thermal_rad == 1:
            print("\t-radiation")
            conv = read_fireca_field("thermalradiation-", qf.ntimes_ave, qf.time_ave, qf, 0, output_folder)
            plot_2d_field(True, qf, plane, 'xy', conv, "Convective heat to human [kW/m^2 skin]", "conv_heat",
                          [], img_specs, no_fuel_idx, flags)
            del conv

        # ------- Energy to atmosphere
        if flags.en2atm == 1:
            print("\t-energy to atmosphere")
            en_to_atm = read_fireca_field("fire-energy_to_atmos-", qf.ntimes, qf.time, qf, 1, output_folder)
            plot_2d_field(False, qf, plane, 'xy', en_to_atm, "Energy to atmosphere [kW/m^3]", "en_to_atm",
                          [], img_specs, [], flags)
            del en_to_atm

        # -------  Fireca winds
        if flags.qf_winds == 1:
            print("\t-fireca winds")
            print("    * u")
            windu_qf = read_fireca_field("windu", qf.ntimes, qf.time, qf, 1, output_folder)
            plot_2d_field(False, qf, plane, 'xy', windu_qf, "U [m/s]", "u", [], img_specs, [], flags)

            print("    * v")
            windv_qf = read_fireca_field("windv", qf.ntimes, qf.time, qf, 1, output_folder)
            plot_2d_field(False, qf, plane, 'xy', windv_qf, "V [m/s]", "v", [], img_specs, [], flags)
            del windv_qf

            print("    * w")
            windw_qf = read_fireca_field("windw", qf.ntimes, qf.time, qf, 1, output_folder)
            plot_2d_field(False, qf, plane, 'xy', windw_qf, "W [m/s]", "w", [], img_specs, [], flags)

        # -------  QU winds (instantaneous)
        if flags.qu_qwinds_inst == 1:
            plane = 2
            vertplane = math.floor(qu.Ly*0.5 / qu.dx)
            print("\t-QU winds")
            print("    * u")
            windu_qu = read_fireca_field("qu_windu", qu.ntimes, qu.time, qu, 1, output_folder)
            plot_2d_field(False, qu, plane, 'xy', windu_qu, "U_inst [m/s]", "u_qu", [], img_specs, [], flags)
            plot_2dvert_field(False, qu, vertplane, 'xz', windu_qu, "U_inst [m/s]",
                              "u_qu_vert", [], img_specs, [], flags)

            print("    * v")
            windv_qu = read_fireca_field("qu_windv", qu.ntimes, qu.time, qu, 1, output_folder)
            plot_2d_field(False, qu, plane, 'xy', windv_qu, "V_inst [vm/s]", "v_qu", [], img_specs, [], flags)
            plot_2dvert_field(False, qu, vertplane, 'xz', windv_qu, "V_inst [vm/s]", "v_qu_vert", [], img_specs, [], flags)
            del windv_qu

            print("    * w")
            windw_qu = read_fireca_field("qu_windw", qu.ntimes, qu.time, qu, 1, output_folder)
            plot_2d_field(False, qu, plane, 'xy', windw_qu, "W_inst [m/s]", "w_qu", [], img_specs, [], flags)
            plot_2dvert_field(False, qu, vertplane, 'xz', windw_qu, "W_inst [m/s]", "w_qu_vert", [], img_specs, [], flags)

        # -------  QU winds (average)
        if flags.qu_qwinds_ave == 1:
            plane = 2
            print("\t-QU winds ave")
            print("    * u")
            windu_qu = read_fireca_field("qu_windu_ave", qu.ntimes_ave, qu.time_ave, qu, 1, output_folder)
            plot_2d_field(True, qu, plane, 'xy', windu_qu, "U_ave [m/s]", "u_qu", [], img_specs, [], flags)

            print("    * v")
            windv_qu = read_fireca_field("qu_windv_ave", qu.ntimes_ave, qu.time_ave, qu, 1, output_folder)
            plot_2d_field(True, qu, plane, 'xy', windv_qu, "V_ave [m/s]", "v_qu", [], img_specs, [], flags)
            del windv_qu

            print("    * w")
            windw_qu = read_fireca_field("qu_windw_ave", qu.ntimes_ave, qu.time_ave, qu, 1, output_folder)
            plot_2d_field(True, qu, plane, 'xy', windw_qu, "W_ave [m/s]", "w_qu", [], img_specs, [], flags)

        # ------- Reaction rate
        plane = 1
        if flags.react_rate == 1:
            print("\t-reaction rate")
            react_rate = read_fireca_field("fire-reaction_rate-", qf.ntimes, qf.time, qf, 0, output_folder)
            plot_2d_field(False, qf, plane, 'xy', react_rate,
                          "Reaction rate [kg/m3/s]", "react_rate", [], img_specs, no_fuel_idx, flags)
            del react_rate

        # ------- Fuel moisture
        if flags.moisture == 1:
            fuels_moist = read_fireca_field("fuels-moist-", qf.ntimes, qf.time, qf, 0, output_folder)
            for iplane in (1, 2, 5, 8, 10, 12):
                if iplane <= qf.nz:
                    print("\t-moisture, plane: %d" % iplane)
                    plot_2d_field(False, qf, iplane, 'xy', fuels_moist,
                                  "Fuel moisture [-]", "fuels_moist", [], img_specs, no_fuel_idx, flags)
            del fuels_moist
    # else:
    #     plot_QU_winds


def import_inputs(prj_folder: str=os.getcwd(), gen_vtk: int=0, gen_gif: int=0):
    print("Importing input data")
    output_folder = os.path.join(prj_folder, 'Output')

    # Initialization
    qu = GridClass()
    qf = GridClass()
    ignitions = IgnitionClass()
    flags = FlagsClass()
    fb = FirebrandClass()

    # Read input files
    print("\t - importing QU grid inputs")
    read_qu_grid(qu, prj_folder)
    print("\t - importing fire inputs")
    read_qfire_file(qf, qu, ignitions, flags, fb, prj_folder, output_folder)
    print("\t - importing terrain elevation")
    read_topo(flags, qu, qf, prj_folder, output_folder)
    print("\t - importing vertical grid details")
    read_vertical_grid(qu, qf, flags, output_folder)
    
    # Setting image specs
    img_specs = ImgClass(gen_gif)
    set_image_specifications(img_specs)

    # Create folder to save images
    create_plots_folder(gen_gif, img_specs, prj_folder)
    
    if flags.isfire == 1:
        print("\t-fuel density field")
        fuel_idx, no_fuel_idx = compress_fuel_dens(qf, flags, output_folder)
    else: fuel_idx = no_fuel_idx = None
    
    return AllDrawFireClasses(qu, qf, ignitions, flags, fb, prj_folder, output_folder, 
                       gen_vtk, gen_gif, img_specs, fuel_idx, no_fuel_idx)


def export_vtk(qf: GridClass, qu: GridClass, flags: FlagsClass):

    # QF
    x = np.arange(0, qf.nx*qf.dx, qf.dx)
    y = np.arange(0, -qf.ny*qf.dy, -qf.dy)
    z = qf.zm

    my_data = {}

    print("Exporting VTK files (fuel)")

    if flags.fuel_density == 1:
        print("\t- add fuel density")
        fuel_dens = read_fireca_field("fuels-dens-", qf.ntimes, qf.time, qf, 0, prj_folder)
        my_data['fuel_density'] = fuel_dens

    if flags.react_rate == 1:
        print("\t- add reaction rate")
        react_rate = read_fireca_field("fire-reaction_rate-", qf.ntimes, qf.time, qf, 0, prj_folder)
        my_data['reaction_rate'] = react_rate

    if flags.en2atm == 1:
        print("\t- add energy to atmos")
        en_to_atm = read_fireca_field("fire-energy_to_atmos-", qf.ntimes, qf.time, qf, 1, prj_folder)
        my_data['energy_to_atmos'] = en_to_atm

    if bool(my_data):
        print('Write fuel VTK files (fuel)')
        for t in range(qf.ntimes):
            print('\ttime: ', qf.time[t])
            out_data = {}
            for k, v in my_data.items():
                out_data[k] = v[t]
            gridToVTK("./fuels-%05d" % qf.time[t], y, x, z, pointData=out_data)

    # QU
    x = np.arange(0, qu.nx * qu.dx, qu.dx)
    y = np.arange(0, -qu.ny * qu.dy, -qu.dy)
    z = qu.zm[1:-1]

    my_data = {}

    if flags.qu_qwinds_inst == 1:
        print("\t- Winds")
        print("\t\t* u")
        windu_qu = read_fireca_field("qu_windu", qu.ntimes, qu.time, qu, 1, prj_folder)

        print("\t\t* v")
        windv_qu = read_fireca_field("qu_windv", qu.ntimes, qu.time, qu, 1, prj_folder)

        print("\t\t* w")
        windw_qu = read_fireca_field("qu_windw", qu.ntimes, qu.time, qu, 1, prj_folder)

        my_data['u'] = windu_qu
        my_data['v'] = windv_qu
        my_data['w'] = windw_qu

    if bool(my_data):
        print('Write Wind VTK files (QU winds)')
        for t in range(qu.ntimes):
            print('\ttime: ', qu.time[t])
            out_data = {
                'winds': (my_data['v'][t], my_data['u'][t], my_data['w'][t]),
                'wind_speed': np.sqrt(np.power(my_data['u'][t], 2) +
                                      np.power(my_data['v'][t], 2) +
                                      np.power(my_data['w'][t], 2))}
            for k, v in my_data.items():
                out_data[k] = v[t]
            gridToVTK("./quwinds-%05d" % qu.time[t], y, x, z, pointData=out_data)


def main(prj_folder: str=os.getcwd(), gen_vtk: int=0, gen_gif: int=0):

    # read input files
    df_classes = import_inputs(prj_folder, gen_vtk, gen_gif)

    # plot outputs
    plot_outputs(df_classes)

    # VTK
    if gen_vtk == 1:
        export_vtk(qf, qu, flags)

    print("Program terminated")

if __name__ == '__main__':
    # Parameters:
    # 1) project folder
    # 2) generate VTK
    # 3) generate GIF
    
    if len(sys.argv) == 1:
        prj_folder_in = os.getcwd()
        gen_vtk_in = 0
        gen_gif_in = 0
    else:
        prj_folder_in = sys.argv[1]

    if len(sys.argv) == 2:        
        gen_vtk_in = 0
        gen_gif_in = 0
    elif len(sys.argv) == 3:
        gen_vtk_in = int(sys.argv[2])
        gen_gif_in = 0
    elif len(sys.argv) == 4:
        gen_vtk_in = int(sys.argv[2])
        gen_gif_in = int(sys.argv[3])

    main(prj_folder_in, gen_vtk_in, gen_gif_in)
