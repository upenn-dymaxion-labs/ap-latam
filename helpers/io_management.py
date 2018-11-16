import os
from glob import glob

def all_raster_files(dirname, ext='.tif'):
    """Generate any raster files inside +dirname+, recursively"""
    dirname = '/Users/dishajindal/Documents/ap-latam/aplatam_data/3band/'
    pattern = '*{ext}'.format(ext=ext)
    return glob(os.path.join(dirname, pattern))