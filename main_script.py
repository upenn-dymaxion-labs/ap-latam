from shapely.geometry import box, shape, mapping
import rasterio
import rasterio.mask
import fiona
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from helpers.io_management import all_raster_files
from helpers.transformations import reproject_shape, getOverlappingShapes, getIntersectionArea, generateWindowsWithMasksForRasters
from helpers.plotting import getShapeCoords, plotShapesOnRaster
from helpers.masking import getBinaryMask, createMask

# Main Script

#e = all_raster_files(dirname = '/Users/dishajindal/Documents/ap-latam/aplatam_data/3band/', 'asd')
#dataset = rasterio.open(e[0])

vector = fiona.open('/Users/dishajindal/Documents/ap-latam/aplatam_data/montevideo_2018_valid.shp')

#north_up = dataset.transform.e <= 0
#rotated = dataset.transform.b != 0 or dataset.transform.d != 0
#matchingShapes = getOverlappingShapes(dataset, vector)
#len(matchingShapes)
#for i in range(len(matchingShapes)):
#    poly = matchingShapes[i][2]
#    fig = plt.figure() 
#    ax = fig.gca() 
#    ax.add_patch(PolygonPatch(poly, fc='#6699cc', ec='#6699cc', alpha=0.5, zorder=2 ))
#    ax.axis('scaled')
#    plt.show()

#matchingShapes = getOverlappingShapes(dataset, vector, returnCoord = True)
#b_mask = getBinaryMask(dataset, matchingShapes)
generateWindowsWithMasksForRasters("/Users/dishajindal/Documents/ap-latam/aplatam_data/3band/", vector)



