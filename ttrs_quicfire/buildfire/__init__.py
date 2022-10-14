"""
TTRS-QF

A simple fire modeling library.
"""

#https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
__all__ = ["build_FF_domain", "build_shapefiles", "buildfire", "dat_file_functions", 
           "exceptions", "print_inp_files"]
__version__= "5.0.0"
__author__= "Zachary Cope"
__credits__="Tall Timbers"

from ttrs_quicfire.buildfire import buildfire