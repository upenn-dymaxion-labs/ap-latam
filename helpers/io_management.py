import os
from glob import glob

"""
Generates any raster files inside the directory, recursively.

Parameters
--------------------
dirname : Directory Path

"""
def all_raster_files(dirname, ext='.tif'):
    pattern = '*{ext}'.format(ext=ext)
    return glob(os.path.join(dirname, pattern))
    