from shapely.geometry import box, shape, mapping
from rasterio.windows import Window
from functools import partial
import pyproj
from shapely.ops import transform

"""
Returns a ploygon object with the coordinates projected from source to destination CRS

Parameters
-------------------- 
shape : shapely.geometry.polygon.Polygon Object
src_crs : CRS of the source system
dst_crs : CRS of the destination system

"""
def reproject_shape(shape, src_crs, dst_crs):
    project = partial(
        pyproj.transform,
        pyproj.Proj(init=src_crs['init']),
        pyproj.Proj(init=dst_crs['init']))
    return transform(project, shape)

"""
Given a raster file and a list of shapes, returns the list of shapes that intersect with the raster file

Parameters
-------------------- 
raster : Raster image. Rasterio object
shapes : Polygon shape files. Fiona object

"""
def getOverlappingShapes(raster,shapes, returnCoord = False, window = None):    
    if window != None:
        fullBox = window
    else:
        fullBox = Window(0, 0, raster.width, raster.height)
    
    fullWindow = box(*raster.window_bounds(fullBox))
    
    matchingShapes = []
    for i in range(len(shapes)):

        t = reproject_shape(shape(shapes[i]['geometry']), shapes.crs, raster.crs)
        intersect = t.intersection(fullWindow)
        if intersect.area > 0 :
            matchingShapes.append([shapes[i], shape(shapes[i]['geometry']),t])
            
    if returnCoord == True:
        #Return only the geospatial coordinates of the transformed shape
        matchingShapes = [mapping(x[2]) for x in matchingShapes]
    
        
    return(matchingShapes)


"""
Given a Window based on a raster and a shape object, returns the ratio/total area of overlap between
the shape and the window

Parameters
-------------------- 
raster : Raster object
window : Rasterio Window object
shapeDict : Dict containing geospatial coordinates
ratio : If true, returns the ratio of intersection area to full window area, else returns intersection area

"""
def getIntersectionArea(raster, window, shapeDict, ratio = True):
    window_box = box( * raster.window_bounds(window))
    
    if(type(shapeDict)==list):
        shp = shape(shapeDict[0])
        
        for i in range(1,len(shapeDict)):
            shp = shp.union(shape(shapeDict[i]))            
    else:        
        shp = shape(shapeDict)
    
    intersection = shp.intersection(window_box)
    
    boxArea = window_box.area
    interArea = intersection.area
    
    if ratio == True:
        return ( interArea / float(boxArea) )
    else:
        return interArea
    