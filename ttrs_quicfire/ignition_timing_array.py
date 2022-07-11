# -*- coding: utf-8 -*-
"""
Created on Thurs Jun 16 2:48:53 2022

@author: Casey Bielefeld
"""
import math
import numpy as np
import sys
import matplotlib.pyplot as plt


class IgnitionClass:
    def __init__(self):
        self.hor_plane = None
        self.flag = None
class SimField:
    def __init__(self):
        self.isfire = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.z = None
        self.zm = None
        self.sim_time = None
        self.ntimes = None
        self.time = None
        self.ntimes_ave = None
        self.time_ave = None
        self.dt = None
        self.dt_print = None
        self.dt_print_ave = None
        self.horizontal_extent = None
       


def read_input(ignitions):
    
    fid = 'ignite.dat'
    
    #Length of file
    fileObj = open(fid, "r") #opens the file in read mode
    Counter = 0
    #read from file
    Content = fileObj.read()
    CoList = Content.split("\n")
    for i in CoList:
        if i:
            Counter += 1
    fileObj.close


    fileObjAgain = open(fid, "r") #opens the file in read mode
    array = []
    
    #need a check
    whichIgnition = fileObjAgain.readline()
    #Get the digit out of the string first line
    ignition_flag = [int(i) for i in whichIgnition.split() if i.isdigit()]
    #assign the digit from the list to a variable
    ign_flag = ignition_flag[0]
    #Assign the ignite class flag to be used in map_ignite
    ignitions.flag = ign_flag
    #Create a string to be appended to if needed
    x = ""
    if ign_flag == 4:
        i = 0
        #skip first 5 lines
        for i in range(0,5):
            fileObjAgain.readline()
            i += 1 
        j = 0
        #Append the next line to a new array
        for j in range(0,Counter - 6):
            x = fileObjAgain.readline()
            addition = list(map(int, x.split()))
            array.append(addition)
            #array.append(x)
            j += 1
        return array
    
    elif ign_flag == 5: 
        i = 0
        #skip first 5 lines
        for i in range(0,5):
            fileObjAgain.readline()
            i += 1 
        j = 0
        #Append the next line to a new array
        for j in range(0,Counter - 6):
            x = fileObjAgain.readline()
            #Difference is float vs int
            addition = list(map(float, x.split()))
            array.append(addition)
            #array.append(x)
            j += 1
        return array
    
def get_line(fid, datatype):
    return split_string(fid.readline(), datatype)

def split_string(s, datatype):
    # http://stackoverflow.com/questions/4289331/python-extract-numbers-from-a-string
    s = s.strip()
    out = []
    for t in s.split():
        try:
            if datatype == 1:
                out.append(int(t))
            else:
                out.append(float(t))
        except ValueError:
            pass    
    return out[0]

def open_file(filename, howto):
    try:
        fid = open(filename, howto)
        return fid
    except IOError:
        print("Error while opening " + filename)
        input("PRESS ENTER TO CONTINUE.")
        sys.exit()


def read_qu_grid(qu):

    # ------- QU_simparams
    fid = open_file('QU_simparams.inp', 'r')
    fid.readline()  # header
    qu.nx = get_line(fid, 1)
    qu.ny = get_line(fid, 1)
    qu.nz = get_line(fid, 1)
    
    
def read_fire_grid(qu, qf):
    fid = open_file('QUIC_fire.inp', 'r')
    i = 0
    for i in range(0, 12):
        fid.readline()  # ! FIRE GRID
        i += 1
        
    qf.nz = get_line(fid, 1)
    ratiox = get_line(fid, 1)
    ratioy = get_line(fid, 1)
    qf.nx = qu.nx * ratiox
    qf.ny = qu.ny * ratioy
    
def map_ignite(list_ignitions, ignitions):
    i = 0
    if ignitions.flag == 4:
        for i in range(0, len(list_ignitions)):
            y_value = list_ignitions[i][1] 
            x_value = list_ignitions[i][0]  
            ignitions.hor_plane[y_value][x_value] = list_ignitions[i][2]
            i += 1
    elif ignitions.flag == 5:
        
        for i in range(0, len(list_ignitions)):
            
            y_value = int(list_ignitions[i][1] / 2) 
            x_value = int(list_ignitions[i][0] / 2) 
            coordinate_one = [x_value, y_value]
            
            #Set first ignition location to first time
            ignitions.hor_plane[y_value][x_value] = list_ignitions[i][4]
            
            #Second set of values
            y_value = int(list_ignitions[i][3] / 2) 
            x_value = int(list_ignitions[i][2] / 2) 
            coordinate_two = [x_value, y_value]
            ignitions.hor_plane[y_value][x_value] = list_ignitions[i][5]
            
            #Total time run
            time_rate = abs(list_ignitions[i][4] - list_ignitions[i][5])
            
            #Distance between coordinates
            dist = math.dist(coordinate_one, coordinate_two)
            x_distance = abs(coordinate_one[0] - coordinate_two[0])
            y_distance = abs(coordinate_one[1] - coordinate_two[1])
            #Cells per second
            j_update = time_rate / dist
            dx = coordinate_two[0] - coordinate_one[0]
            dy = coordinate_two[1] - coordinate_one[1]
           
            j = list_ignitions[i][4]
            
            #Adapt the returned generator into a list of tuples
            cell_ignitions = list(bresenham(coordinate_one[0], coordinate_one[1], coordinate_two[0], coordinate_two[1], dx, dy, ignitions))
            #Accesed through this method https://www.geeksforgeeks.org/python-get-first-element-in-list-of-tuples/
            for i in cell_ignitions:
                ignitions.hor_plane[i[1]][i[0]] = j
                j += int(round(j_update))
            
    return

#Formatted from https://github.com/encukou/bresenham, returns a generator of 
#the coordinates of the line from (x0, y0) to (x1, y1)
def bresenham(x0, y0, x1, y1, dx, dy, ignitions): 
    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1
    
    dx = abs(dx)
    dy = abs(dy)
    
    if dx > dy :
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0
    D = 2 * dy - dx
    y = 0
    
    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy
    
    
def main():
   #Create Classes
   ignitions = IgnitionClass()
   qu = SimField()
   qf = SimField()
   
   #Read ignite.dat into an array
   list_ignitions = read_input(ignitions)
   
   #Get size of total array
   read_qu_grid(qu)
   read_fire_grid(qu, qf)
   
   #Set an array of set size to values of -1 
   ignitions.hor_plane = np.zeros((qf.ny, qf.nx)) - 1
   
   #Map the ignited points to the array
   map_ignite(list_ignitions, ignitions)
   
   #ignitions.hor_plane is the array 
   #Uncomment to show graph of plotted cells
   #plt.imshow(ignitions.hor_plane,origin='lower')
   #plt.clim(0.1,0.2)

   

if __name__ == '__main__':    
    main()