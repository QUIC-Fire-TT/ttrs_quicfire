# # # # # # #
# D R A F T #
# # # # # # #

# ign_module_v2.py
# Establishes a module of functions to support Rx ignitions
# Various ignition patterns: contours, lines (strips), rings
# Ignition sequencing according to (Hu and Ge 2021)

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import re
import sys
import os
from os.path import exists
from shapely.geometry import Point, LineString, Polygon, mapping

# Class 1 - Contains all relevant settings
#ajustes = {}

# START, function init_shp - Read in shapefiles, initialize data dictionaries
def init_shp(shpdir, burnUnitNames, global_epsg, ign_style, ign_file_names, buf_meth):
    # Initialize dictionaries across burn units
    dict_roads = {}; dict_bbox = {}; dict_ign = {}
    # Loop through burn units, watch out: temporary storage variables are used!
    for nn in range(len(burnUnitNames)):
        # Build name of roads file
        fnam = '{0}roads_{1}.shp'.format(shpdir, burnUnitNames[nn])
        # Check for roads in shpdir
        if exists(fnam):
            # Load and store roads
            roads = gpd.read_file(fnam)
            dict_roads[nn] = roads.to_crs(epsg=global_epsg)
        else:
            print('No roads found')
        # Load and store bounding box(es)
        fnam = '{0}bbox_{1}.shp'.format(shpdir, burnUnitNames[nn])
        # Check for bbox in shpdir
        if exists(fnam):
            bbox = gpd.read_file(fnam)
            dict_bbox[nn] = bbox.to_crs(epsg=global_epsg)
        # Default to a single bounding box
        else:
            bbox = gpd.read_file('{0}bbox.shp'.format(shpdir))
            # And this will be replicated for all units
            dict_bbox[nn] = bbox.to_crs(epsg=global_epsg)
        # Python switch statement on ign_style
        if ign_style == 'ring':
            # Load and store ignition rings according to buf_meth
            ring_ign = gpd.read_file('{0}ring_{1}_{2}.shp'.format(shpdir, burnUnitNames[nn], buf_meth))
            dict_ign[nn] = ring_ign.to_crs(epsg=global_epsg)
        elif ign_style == 'ct':
            # Load and store contour lines
            ct_ign = gpd.read_file('{0}ct_{1}.shp'.format(shpdir, burnUnitNames[nn]))
            dict_ign[nn] = ct_ign.to_crs(epsg=global_epsg)
        elif ign_style == 'strip':
            # TODO: functionality for lines here!!!
            print('Future functionality for {0}'.format(os.path.basename(__file__)))
            sys.exit(0)
        elif ign_style == 'custom':
            # Load and store custom ignition lines
            cust_ign = gpd.read_file('{0}{1}.shp'.format(shpdir, ign_file_names[nn]))
            dict_ign[nn] = cust_ign.to_crs(epsg=global_epsg)
        else:
            print("Don't recognize ignition style ign_style = {0}".format(ign_style))
            print("Check settings in script")
            sys.exit(0)
    return dict_roads, dict_bbox, dict_ign
# END, function init_shp

# START, function ign_seq_to_csv - Construct a pandas dataframe to set ignition sequencing
def ign_seq_to_csv(burnUnitNames, ign_style, ring_thetas, dict_ign, numCohorts, btw_cohort_s, in_cohort_s, csv_fnam_root):
    # For each RxUnit
    for nn in range(len(burnUnitNames)):
        # Initialize empty dataframe with five (5) columns
        df_ign = pd.DataFrame(columns=['id',
                                       'Direction',
                                       'Length_m',
                                       'Ig_Name',
                                       'Add_Time'])
        df_ign = df_ign.astype(dtype={'id': int,
                                      'Direction': str,
                                      'Length_m': float,
                                      'Ig_Name': str,
                                      'Add_Time': float})
        # Python switch statement on ign_style
        if ign_style == 'ring':
            # Incorporate partial rings too (governed by spread in ring_thetas)
            ig_lines = dict_ign[nn]
            # Clip ring down to extent specified by ring_thetas
            line_geo = ig_lines.iloc[0]
            line_geo = line_geo['geometry']
            x0 = line_geo.centroid
            # Compute and pad max distance from centroid to ring
            r_outer = np.ceil(x0.hausdorff_distance(line_geo))
            # Compute and pad min distance from centroid to ring
            r_inner = np.floor(x0.distance(line_geo))
            # Set up fine mesh for scan wedge
            n_points = 128
            angles = np.linspace(ring_thetas[0], ring_thetas[1], n_points)
            if np.abs(ring_thetas[1] - ring_thetas[0]) == 360.0:
                # Disconnect the ring (avoids self-intersections that crash the clip routine later on)
                #line_geo = LineString(line_geo.coords[1:len(line_geo.coords)])
                # Once again, slightly disconnecting annulus helps a lot
                angles = angles[:-1]
            # Build a scan wedge polygon
            scannulus_outer = [[r_outer*np.cos(theta*np.pi/180.0)+x0.coords[0][0],
                                r_outer*np.sin(theta*np.pi/180.0)+x0.coords[0][1]] for theta in angles]
            scannulus_inner = [[r_inner*np.cos(theta*np.pi/180.0)+x0.coords[0][0],
                                r_inner*np.sin(theta*np.pi/180.0)+x0.coords[0][1]] for theta in angles]
            scannulus = Polygon(np.vstack([scannulus_outer, scannulus_inner]))
            # Convert to GeoDataFrames to use clip tool
            line_geo = gpd.GeoDataFrame(index=[0], geometry=[line_geo])
            scannulus = gpd.GeoDataFrame(index=[0], geometry=[scannulus])
            # Clip the line ignition to teh scan_wedge
            partial_ring = gpd.clip(line_geo, scannulus)

            # Diagnostic plotting
            fig0, axs0 = plt.subplots(2)
            ig_lines.plot(ax=axs0[0])
            axs0[0].scatter(x0.coords[0][0], x0.coords[0][1], s=12.0, c='b')
            scannulus.plot(ax=axs0[0], alpha=0.44, color='r')
            axs0[1].scatter(x0.coords[0][0], x0.coords[0][1], s=12.0, c='b')
            partial_ring.plot(ax=axs0[1])
            scannulus.plot(ax=axs0[1], alpha=0.44, color='g')
            plt.show()
            
            # Overwrite ignition with this ring
            dict_ign[nn] = partial_ring
            # id
            df_ign['id'] = range(1, len(ig_lines)+1)
            # Direction
            df_ign['Direction'] = 'W-E'
            # Length_m
            df_ign['Length_m'] = ig_lines.length[0]
            # Organize the ignition sequencing into cohorts
            maxlinespercohort = int(np.ceil(len(ig_lines) / numCohorts))
            # Loop to name the ignition line within cohort
            # Also build a vector for Add_Time (seconds)
            i = 0
            ig_name_list = []  # Empty list for ignition names (line # within cohort)
            add_time_vec = [0.0]  # Initial add_time is 0.0 (seconds)
            for nc in range(numCohorts):
                if nc > 0:
                    # Append a between cohort delay time to Add_Time_vec
                    add_time_vec.append(btw_cohort_s)
                for nl in range(maxlinespercohort):
                    # Append the string cohort-lineWithinCohort to the Ig_Name_list
                    ig_name_list.append('{0}-{1}'.format(nc + 1, nl + 1))
                    # If the next iterate strides out of range
                    if (i + 1 >= len(ig_lines)):
                        # Break the inner for loop
                        break
                    # If the next iterate is acceptable
                    else:
                        if nl < maxlinespercohort - 1:
                            # Append a within cohort delay time to Add_Time_vec
                            add_time_vec.append(in_cohort_s)
                        # Iterate
                        i += 1
                        continue
                    # If the inner loop is broken, then the outer loop breaks too
                    break
            # Ig_Name was build within the above loop
            df_ign['Ig_Name'] = ig_name_list
            # Add_Time was built within the above loop
            df_ign['Add_Time'] = add_time_vec
            # Save this dataframe to a csv file
            df_ign.to_csv('{0}{1}.csv'.format(csv_fnam_root, nn + 1), index=False)
        elif ign_style == 'ct' or ign_style == 'strip':
            # Contour ignition lines or strips work here
            ig_lines = dict_ign[nn]
            # id
            df_ign['id'] = range(1, len(ig_lines)+1)
            # Loop to set preferred directions
            direction_list = []
            for i in range(len(ig_lines)):
                # If odd
                if np.mod(i, 2):
                    # Suggest N-S ignition direction
                    direction_list.append('E-W')
                # Else even
                else:
                    # Suggest S-N ignition direction
                    direction_list.append('W-E')
            # Direction
            df_ign['Direction'] = direction_list
            # Length_m
            df_ign['Length_m'] = ig_lines.Shape_Leng
            # Organize the ignition sequencing into cohorts
            maxlinespercohort = int(np.ceil(len(ig_lines) / numCohorts))
            # Loop to name the ignition line within cohort
            # Also build a vector for Add_Time (seconds)
            i = 0
            ig_name_list = []  # Empty list for ignition names (line # within cohort)
            add_time_vec = [0.0]  # Initial add_time is 0.0 (seconds)
            for nc in range(numCohorts):
                if nc > 0:
                    # Append a between cohort delay time to Add_Time_vec
                    add_time_vec.append(btw_cohort_s)
                for nl in range(maxlinespercohort):
                    # Append the string cohort-lineWithinCohort to the Ig_Name_list
                    ig_name_list.append('{0}-{1}'.format(nc+1, nl+1))
                    # If the next iterate strides out of range
                    if (i + 1 >= len(ig_lines)):
                        # Break the inner for loop
                        break
                    # If the next iterate is acceptable
                    else:
                        if nl < maxlinespercohort - 1:
                            # Append a within cohort delay time to Add_Time_vec
                            add_time_vec.append(in_cohort_s)
                        # Iterate
                        i += 1
                        continue
                    # If the inner loop is broken, then the outer loop breaks too
                    break
            # Ig_Name was build within the above loop
            df_ign['Ig_Name'] = ig_name_list
            # Add_Time was built within the above loop
            df_ign['Add_Time'] = add_time_vec
            # Save this dataframe to a csv file
            df_ign.to_csv('{0}{1}.csv'.format(csv_fnam_root, nn+1), index=False)
        elif ign_style == 'custom':
            # TODO: Consider multiple ignition files per burn unit (for now only one)
            ig_lines = dict_ign[nn]
            # id
            df_ign['id'] = range(1, len(ig_lines) + 1)
            # Direction
            df_ign['Direction'] = ['W-E']
            # Length_m
            df_ign['Length_m'] = ig_lines.length[0]
            # Organize the ignition sequencing into cohorts
            maxlinespercohort = int(np.ceil(len(ig_lines) / numCohorts))
            # Loop to name the ignition line within cohort
            # Also build a vector for Add_Time (seconds)
            i = 0
            ig_name_list = []  # Empty list for ignition names (line # within cohort)
            add_time_vec = [0.0]  # Initial add_time is 0.0 (seconds)
            for nc in range(numCohorts):
                if nc > 0:
                    # Append a between cohort delay time to Add_Time_vec
                    add_time_vec.append(btw_cohort_s)
                for nl in range(maxlinespercohort):
                    # Append the string cohort-lineWithinCohort to the Ig_Name_list
                    ig_name_list.append('{0}-{1}'.format(nc + 1, nl + 1))
                    # If the next iterate strides out of range
                    if (i + 1 >= len(ig_lines)):
                        # Break the inner for loop
                        break
                    # If the next iterate is acceptable
                    else:
                        if nl < maxlinespercohort - 1:
                            # Append a within cohort delay time to Add_Time_vec
                            add_time_vec.append(in_cohort_s)
                        # Iterate
                        i += 1
                        continue
                    # If the inner loop is broken, then the outer loop breaks too
                    break
            # Ig_Name was build within the above loop
            df_ign['Ig_Name'] = ig_name_list
            # Add_Time was built within the above loop
            df_ign['Add_Time'] = add_time_vec
            # Save this dataframe to a csv file
            df_ign.to_csv('{0}{1}.csv'.format(csv_fnam_root, nn + 1), index=False)
        else:
            print("Don't recognize ignition style ign_style = {0}".format(ign_style))
            print("Check settings in script")
            sys.exit(0)
    return df_ign
# END, function ign_seq_to_csv

# START, function build_ig_lines - Create ignition points along the lines
def build_ig_lines(burnUnitNames, dict_ign, distance_delta, points_fnam_root, ign_style):
    dict_ig_points = {}
    for nn in range(len(burnUnitNames)):
        ig_lines = dict_ign[nn]
        dict_ig_points[nn] = gpd.GeoDataFrame()
        for i in range(len(ig_lines)):
            line = ig_lines.iloc[i]
            line_geo = line['geometry']
            distances = np.arange(0, line_geo.length, distance_delta)
            # Treat rings and contours slightly different here
            if ign_style == 'ct':
                # Catch for empty line_geo (closed contours)
                if bool(line_geo.boundary):
                    points = [line_geo.interpolate(distance) for distance in distances] + [line_geo.boundary.geoms[1]]
                    temp_dict = {'geometry': points}
                    for k in line.keys():
                        if k != 'geometry':
                            temp_dict[k] = [line[k], ] * len(points)
                    temp_points = gpd.GeoDataFrame(temp_dict, crs=5070)
                    dict_ig_points[nn] = dict_ig_points[nn].append(temp_points)
            elif ign_style == 'ring':
                # There will be no boundary, interpolate from initial point
                points = [line_geo.interpolate(distance) for distance in distances]
                temp_dict = {'geometry': points}
                for k in line.keys():
                    if k != 'geometry':
                        temp_dict[k] = [line[k], ] * len(points)
                temp_points = gpd.GeoDataFrame(temp_dict, crs=5070)
                dict_ig_points[nn] = dict_ig_points[nn].append(temp_points)
        # Save to file
        fnam = "{0}{1}".format(points_fnam_root, nn+1)
        dict_ig_points[nn].to_file(fnam)
    return dict_ig_points
# END, function build_ig_lines

# START, function build_ignite_dot_dat - Build ignite.dat files for QUIC-fire simulations
def write_ignite_dot_dat(burnUnitNames, outdir, dict_Ig_Points, csv_fnam_root, dict_bbox, strt_time, SPEED_OF_IGNITION, res_xy_m):
    # For each RxUnit
    for nn in range(len(burnUnitNames)):
        # Name QUIC-fire ignition file
        fnam = "{0}ignite0{1}.dat".format(outdir, nn+1)
        # Empty dictionary to start with the header of ignite.dat
        LINES_2_WRITE = {}
        # Aerial type ignitions give the most flexibility for point ignitions
        LINES_2_WRITE[0] = "       igntype=    4		!Ignitions flag: 4 = Aerial ignition, 5 = ATV ignition\n"
        # Indicates there will be a list of geolocations as the firing points
        LINES_2_WRITE[1] = "    &aeriallist\n"
        # Write the total number of point ignitions for simulation
        LINES_2_WRITE[2] = "       naerial=  {0}		!Number of point ignitions in simulation\n".format(len(dict_Ig_Points[nn]))
        LINES_2_WRITE[3] = "    targettemp= 1000.00		!DONT CHANGE: Ignition temperature\n"
        LINES_2_WRITE[4] = "      ramprate=  172.00\n"
        LINES_2_WRITE[5] = "/				!This line should be followed by naerial rows specifying the ignition attributes. If naerial=5 the code will expect 5 rows.\n"
        # This list of lines will grow via iteration from here
        ll = 6
        # Load ignition sequence for current unit from a csv file
        df_ign = pd.read_csv('{0}{1}.csv'.format(csv_fnam_root, nn+1))
        # Bounding box coordinates
        x_min, y_min, x_max, y_max = dict_bbox[nn].total_bounds
        # Loop through each line ignition present in this dataframe
        for ig in range(len(df_ign)):
            # Current line
            curr_line = df_ign[df_ign['id'] == (ig + 1)]
            # Update running ignition clock
            strt_time += float(curr_line['Add_Time'])
            # Compute time interval in which to finish this ignition
            time_interval = float(curr_line['Length_m']) / SPEED_OF_IGNITION
            # Load ignition points for current line from dictionary
            if hasattr(dict_Ig_Points, 'OBJECTID'):
                # Might be indexed by OBJECTID = ig + 1
                Ig_Points = dict_Ig_Points[nn].loc[dict_Ig_Points[nn].OBJECTID == (ig + 1), :]
            else:
                # Otherwise could be a singleton
                Ig_Points = dict_Ig_Points[nn]
            # Check for empty Ig_Points
            # TODO: May need to add a catch for closed contours here as well!!!!!!!!!!!!
            if len(Ig_Points) > 0:
                # Compute time step for clock ticks from one ignition point to next
                time_step = time_interval / len(Ig_Points)
                # Convert current line of points to QUIC-fire grid cell addresses
                xvec = list(round((Ig_Points.geometry.x - x_min) / res_xy_m[0]))
                yvec = list(round((Ig_Points.geometry.y - y_min) / res_xy_m[1]))
                # Switch on ignition direction 'N-S' , 'S-N' , 'E-W' , 'W-E'
                ig_dir = curr_line['Direction'].unique()[0]
                if (ig_dir == 'N-S'):
                    # Decide which endpoint of line is more northerly
                    if yvec[0] > yvec[-1]:  # Initial point is more northerly
                        xx = xvec.pop(0)
                        yy = yvec.pop(0)
                    else:  # Terminal point is more northerly
                        xx = xvec.pop(-1)
                        yy = yvec.pop(-1)
                elif (ig_dir == 'S-N'):
                    # Decide which endpoint of line is more southerly
                    if yvec[0] > yvec[-1]:  # Terminal point is more southerly
                        xx = xvec.pop(-1)
                        yy = yvec.pop(-1)
                    else:  # Initial point is more southerly
                        xx = xvec.pop(0)
                        yy = yvec.pop(0)
                elif (ig_dir == 'W-E'):
                    # Decide which endpoint of line is more westerly
                    if xvec[0] > xvec[-1]:  # Terminal point is more westerly
                        xx = xvec.pop(-1)
                        yy = yvec.pop(-1)
                    else:   # Initial point is more westerly
                        xx = xvec.pop(0)
                        yy = yvec.pop(0)
                elif (ig_dir == 'E-W'):
                    # Decide which endpoint is more easterly
                    if xvec[0] > xvec[-1]:  # Initial point is more easterly
                        xx = xvec.pop(0)
                        yy = yvec.pop(0)
                    else:   # Terminal point is more easterly
                        xx = xvec.pop(-1)
                        yy = yvec.pop(-1)
                else:
                    print("Don't recognize ignition direction: {}".format(ig_dir))
                    print("Check CSV")
                    sys.exit(0)
                # Loop through ignition points
                for j in range(len(xvec)):
                    # Compute time clock reading for this ignition point
                    strt_time += time_step
                    # Write ignition location and timing
                    LINES_2_WRITE[ll] = '{0}  {1}    {2:.2f}\n'.format(int(xx), int(yy), strt_time)
                    # Iterate to next line
                    ll += 1
                    # Determine nearest neighbor
                    diff = np.vstack((xvec, yvec)) - np.vstack((np.ones((1, len(xvec)))*xx, np.ones((1, len(yvec)))*yy))
                    diff2 = diff*diff
                    sq_diff = np.sqrt(sum(diff2, 1))
                    next_idx = np.argmin(sq_diff)
                    xx = xvec.pop(next_idx)
                    yy = yvec.pop(next_idx)
        # Check for something beyond the header of the ignition file
        if len(LINES_2_WRITE) > 9:
            # Add an informative comment next to first line of ignition cell locations
            LINES_2_WRITE[6] = LINES_2_WRITE[6].replace('\n', '\t\t!x_location [cell number], y_location [cell number], ignition time [s]\n')
            # Add a comment for a useful example
            LINES_2_WRITE[8] = LINES_2_WRITE[8].replace('\n', '\t\t!EXAMPLE: Assuming cell dimentions are 2x2m, to start a point ignition at x = 100m, y = 100m, time = 5 seconds. Row should look as follows:\n')
            LINES_2_WRITE[9] = LINES_2_WRITE[9].replace('\n', '\t\t!  50  50    5.0\n')
        # Write to ignite.dat file
        with open(fnam, 'w') as infp:
            for ll in range(len(LINES_2_WRITE)):
                infp.write(LINES_2_WRITE[ll])
# END, function write_ignite_dot_dat

# START, funSction read_ignite_dot_dat - Routine to read ignite.dat style files directly
def read_ignite_dot_dat(fnam):
    with open(fnam) as fp:
        lines = fp.read().splitlines()
    # Initialize list of ignition points
    xyt = []
    # Read locations and timings
    for ll in range(6, len(lines)):
        xyt.append(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", lines[ll]))
    return xyt
# END, function read_ignite_dot_dat

# START, function plot_ig - Build a 2x2 plot array to check configurations
def plot_ig(burnUnitNames, dict_roads, dict_bbox, shpdir, dict_ign, dict_Ig_Points, res_xy_m, outdir):
    # Learn number of burn units
    num_units = len(burnUnitNames)
    # 2x2 subplots
    fig1, axs1 = plt.subplots(2, 2)
    # Top left plot
    for nn in range(num_units):
        # Check for roads
        if not dict_roads:
            print('No roads plotted')
        else:
            dict_roads[nn].plot(alpha=0.9, color='tab:brown', ax=axs1[0, 0])
        dict_bbox[nn].boundary.plot(alpha=0.5, color='slategray', ax=axs1[0, 0])
    axs1[0, 0].set_title('{0}roads.shp & bboxes'.format(shpdir))
    # Top right plot
    for nn in range(num_units):
        # Check for roads
        if not dict_roads:
            print('No roads plotted again')
        else:
            dict_roads[nn].plot(alpha=0.5, color='tab:brown', ax=axs1[0, 1])
    for nn in range(num_units):
        dict_ign[nn].plot(alpha=0.9, color='tab:orange', ax=axs1[0, 1])
    axs1[0, 1].set_title('Contours')
    # Bottom left plot
    for nn in range(num_units):
        dict_Ig_Points[nn].plot(alpha=0.9, marker='.', markersize=2.5, color='tab:orange', ax=axs1[1, 0])
    axs1[1, 0].set_title('Planned Ignition Points')
    # Bottom right plot
    #nn = 0
    for nn in range(num_units):
        x_min, y_min, x_max, y_max = dict_bbox[nn].total_bounds
        # Check for roads
        if not dict_roads:
            print('No roads plotted yet again')
        else:
            roads2plot = dict_roads[nn].translate(xoff=-x_min, yoff=-y_min, zoff=0.0)
            roads2plot = roads2plot.scale(xfact=1.0 / res_xy_m[0], yfact=1.0 / res_xy_m[1], zfact=1.0, origin=(0.0, 0.0))
            roads2plot.plot(alpha=0.5, color='tab:brown', ax=axs1[1, 1])
        xy_contours = dict_ign[nn].translate(xoff=-x_min, yoff=-y_min, zoff=0.0)
        xy_contours = xy_contours.scale(xfact=1.0 / res_xy_m[0], yfact=1.0 / res_xy_m[1], zfact=1.0, origin=(0.0, 0.0))
        xy_contours.plot(alpha=0.8, color='k', linewidth=0.68, ax=axs1[1, 1])
        xyt = read_ignite_dot_dat('{0}ignite0{1}.dat'.format(outdir, nn+1))
        hsv = plt.cm.get_cmap('hsv', len(xyt))
        newcolors = hsv(np.linspace(0, 1, len(xyt)))
        for pp in range(len(xyt)):
            axs1[1, 1].plot(int(xyt[pp][0]), int(xyt[pp][1]), color='k', marker='s', markersize=8.0)
            num_mrkr = '${0}$'.format(str(pp + 1))
            axs1[1, 1].plot(int(xyt[pp][0]), int(xyt[pp][1]), color=newcolors[pp], marker=num_mrkr, markersize=7.0)
        axs1[1, 1].set_xlabel('Easting (cell #)')
        axs1[1, 1].set_ylabel('Northing (cell #)')
        axs1[1, 1].set_title('Planned Ignition Sequence')
    plt.show()
# END, function plot_ig

# START, function linestring_to_polygon - Change from LineString to polygon
def linestring_to_polygon(gdf):
    #https://stackoverflow.com/questions/2964751/how-to-convert-a-geos-multilinestring-to-polygon
    geom = [x for x in gdf.geometry]
    all_coords = mapping(geom[0])['coordinates']
    lats = [x[1] for x in all_coords]
    lons = [x[0] for x in all_coords]
    polyg = Polygon(zip(lons, lats))
    return gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[polyg])
# END, function linestring_to_polygon

# START, function config_strip_ig - Configure strip ignitions
def config_strip_ig(bbox0, theta0, nlines, global_epsg, burnplt):
    # Extent of original bounding box, origin for coord sys @ (xmin0,ymin0)
    xmin0, ymin0, xmax0, ymax0 = bbox0.total_bounds
    # Apply rotation because ignition line code works in the first quadrant
    # i.e. 0 <= theta < 90
    if theta0 > 90.0 and theta0 <= 180.0:
        # Rotate bounding box by 90 cw about (xmin0, ymin0)
        bbox = bbox0.rotate(-90.0, origin=(xmin0, ymin0))
        theta = theta0 - 90.0
    elif theta0 > 180 and theta0 <= 270.0:
        # Rotate bounding box by 180 cw about (xmin0, ymin0)
        bbox = bbox0.rotate(-180.0, origin=(xmin0, ymin0))
        theta = theta0 - 180.0
    elif theta0 > 270.0 and theta0 <= 360.0:
        # Rotate bounding box by 270 cw about (xmin0, ymin0)
        bbox = bbox0.rotate(-270.0, origin=(xmin0, ymin0))
        theta = theta0 - 270.0
    else:
        # No rotation necessary
        bbox = bbox0
        theta = theta0
    # Extent of working bounding box (potentially rotated)
    xmin, ymin, xmax, ymax = bbox.total_bounds
    # Line spacing based on number of gridlines (aim to set this directly later)
    rho = min((xmax - xmin) / nlines, (ymax - ymin) / nlines)
    # Set fine mesh resolution
    nmesh = 128
    # Establish fine mesh on bbox, calibrated to diagonal length
    rmesh = np.sqrt((ymax - ymin) ** 2 + (xmax - xmin) ** 2) / nmesh

    # Algorithm for generating lines on the fine mesh
    # Store all ignition lines in long list
    routes = {}  # Will be a list of GeoDataFrames
    # Iterator through routes
    ii = 0
    # Horizontal lines do not need the first loop at all
    if np.sin(theta * np.pi / 180.0) > 0.10452846326765346:
        # First loop for drawing lines RIGHT from lower-left corner (xmin, ymin)
        xstart = xmin + (rho / np.sin(theta * np.pi / 180.0))
        # While the starting point lies along the LOWER EDGE of bbox
        while xstart <= xmax:
            # Build line ignition across bbox at angle theta
            xx = xstart
            # Always start these lines along LOWER EDGE of bbox
            yy = ymin
            xline = []  # Will be built up with x-coords
            yline = []  # Will be built up with y-coords
            geom = []  # Stores point geometries (xx, yy)
            # Inner loop to set points along fine mesh
            while (xx <= xmax) and (yy <= ymax):
                # Add x-coords, y-coords, and point geometries to growing list
                xline.append(xx)
                yline.append(yy)
                geom.append(Point(xx, yy))
                # Iterate a distance rmesh across the fine mesh in direction theta
                xx = xx + rmesh * np.cos(theta * np.pi / 180.0)
                yy = yy + rmesh * np.sin(theta * np.pi / 180.0)
            # Build a pandas dataframe with line coordinates
            d = {'col1': np.ones_like(xline), 'col2': xline, 'col3': yline}
            # Print to text file (for diagnostics)
            with open("temp.txt", 'w') as f:
                for key, value in d.items():
                    f.write('%s:%s\n' % (key, value))
            # As long as there is more than one point to work with
            if len(geom) > 1:
                # Draw a LineString
                line = LineString(geom)
                # Build and add this geometry to the routes list
                d = {'geometry': [line]}
                routes[ii] = gpd.GeoDataFrame(d, epsg=global_epsg)
                print('...routes[{0}] written to memory'.format(ii))
                # Iterate to next position on routes list
                ii = ii + 1
            # Next line to the RIGHT
            xstart = xstart + (rho / np.sin(theta * np.pi / 180.0))
        # De-iterate (to know where to store lines from the next loop)
        ii = ii - 1
    # Vertial lines do not need the second loop at all
    if np.cos(theta * np.pi / 180.0) != 0.0:
        # Second loop for drawing lines UP from lower-left corner (xmin, ymin)
        ystart = ymin
        # Iterator through routes
        jj = 0
        # While the starting point lies along the LEFT EDGE of bbox
        while ystart <= ymax:
            # Always start along LEFT EDGE of bbox
            xx = xmin
            # Build line ignition across bbox at angle theta
            yy = ystart
            xline = []  # Will be built up with x-coords
            yline = []  # Will be built up with y-coords
            geom = []  # Stores point geometries (xx, yy)
            # Inner loop to set points along fine mesh
            while (xx <= xmax) and (yy <= ymax):
                # Add x-coords, y-coords, and point geometries to growing list
                xline.append(xx)
                yline.append(yy)
                geom.append(Point(xx, yy))
                # Iterate a distance rmesh across the fine mesh in direction theta
                xx = xx + rmesh * np.cos(theta * np.pi / 180.0)
                yy = yy + rmesh * np.sin(theta * np.pi / 180.0)
            # Build a pandas dataframe with line coordinates
            d = {'col1': np.ones_like(xline), 'col2': xline, 'col3': yline}
            # As long as there is more than one point to work with
            if len(geom) > 1:
                # Draw a LineString
                line = LineString(geom)
                # Build and add this geometry to the routes list
                d = {'geometry': [line]}
                routes[ii + jj] = gpd.GeoDataFrame(d, epsg=global_epsg)
                print('...routes[{0}] written to memory'.format(ii + jj))
                # Iterate to next position on routes list
                jj = jj + 1
            # Next line UP
            ystart = ystart + (rho / np.cos(theta * np.pi / 180.0))
    # Report size of routes to the user (for diagnostics)
    print('__sizeof__ routes is {0}'.format(routes.__sizeof__()))

    # Apply the opposite rotation as was initially applied to the original bounding box
    routes0 = {}
    if theta0 > 90.0 and theta0 <= 180.0:
        # Loop through routes
        for kk in range(len(routes)):
            # Rotate the ignition lines back to the original frame
            routes0[kk] = routes[kk].rotate(90.0, origin=(xmin0, ymin0))
    elif theta0 > 180.0 and theta0 <= 270.0:
        # Loop through routes
        for kk in range(len(routes)):
            # Rotate the ignition lines back to the original frame
            routes0[kk] = routes[kk].rotate(180.0, origin=(xmin0, ymin0))
    elif theta0 > 270.0 and theta0 <= 360.0:
        # Loop through routes
        for kk in range(len(routes)):
            # Rotate the ignition lines back to the original frame
            routes0[kk] = routes[kk].rotate(-90.0, origin=(xmin0, ymin0))
    else:
        # Loop through routes
        for kk in range(len(routes)):
            # No rotation necessary
            routes0[kk] = routes[kk]

    # Clip routes to the burn plot
    routes0_clip = {}
    for kk in range(len(routes0)):
        routes0_clip[kk] = gpd.clip(routes0[kk], burnplt)

    # Reorganize into a singleton dictionary
    Lines = {'geometry': []}
    for kk in range(len(routes0)):
        Lines['geometry'].append(list(routes0[kk].geometry))
    Lines['geometry'] = [val for sublist in Lines['geometry'] for val in sublist]
    ignition = pd.DataFrame(Lines)
    ignition = gpd.GeoDataFrame(ignition, geometry=ignition.geometry, crs=5070)
    ignition = gpd.clip(ignition, burnplt)

    return routes0_clip, ignition
# END, function config_strip_ig
