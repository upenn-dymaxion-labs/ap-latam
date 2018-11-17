import matplotlib.pyplot as plt

"""
Returns the pixel positions of the the shapes given in geospatial coordinates using the indexing from raster object.
This allows the plotting of shapes on raster images

Parameters
--------------------  
raster : Image file. Rasterio object
shape : Dict containing polygons in geospatial coordinates

"""
def getShapeCoords(raster, shape, returnXY = False):
    coordList = []

    for i in range(len(shape['coordinates'][0])):
        point = shape['coordinates'][0][i]
        newPoint = raster.index(point[0], point[1])
        coordList.append(newPoint)
        
    #This step is done because the ordering of x,y axis is reversed while using index.
    #This behavior has been observed at different instances across rasterio
    coordList = [(x[1],x[0]) for x in coordList]
    
    if returnXY == True:
        x = [i[0] for i in coordList]
        y = [i[1] for i in coordList]
        return(x,y)
    else:
        return(coordList)
        
"""
Plots the raster object and the polygons on top of it from the shapes object. This is done by finding 
the corresponding pixel positions of the shapes by calling getShapeCoords method.

Parameters
-------------------- 
raster : Raster file object
shapes : Dict containing polygons in geospatial coordinates
band : band of the raster to be used

"""
def plotShapesOnRaster(raster, shapes, band = 1):
    rval = raster.read(band)
    plt.figure(figsize=(20,20))
    plt.imshow(rval, cmap='gray')
    for i in range(len(shapes)):
        shape = shapes[i];
        shape_coords = getShapeCoords(raster, shape, returnXY=True)
        plt.plot(shape_coords[0], shape_coords[1], color='red', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
