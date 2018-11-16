import matplotlib.pyplot as plt

def getShapeCoords(raster, shape,returnXY = False):
    """
    Given a raster and a shape ( polygon coordinates), returns the polygon coordinates in terms
    of pixel positions.
    This allows the plotting of shapes on raster images
    
    Arguments 
    raster : Image file. Rasterio object
    shape : Dict containing geospatial coordinates
    """
    
    coordList = []
    
    for i in range(len(shape['coordinates'][0])):
        point = shape['coordinates'][0][i]
        newPoint = raster.index(point[0], point[1])
        
        coordList.append(newPoint)
        
    #This step is done because the ordering of x,y axis is reversed while using index.
    #This behavior has been observed at different instances across rasterio
    coordList = [(x[1],x[0]) for x in coordList]
    
    if returnXY==True:
        x = [i[0] for i in coordList]
        y = [i[1] for i in coordList]
        return(x,y)
    else:
        return(coordList)
        

def plotShapesOnRaster(raster, shapes, band = 1):
    """
    
    Arguments :
    raster : Raster file object
    shape : Dict containing geospatial coordinates
    band : Which band of the raster to be used
    """
    
    rval = raster.read(band)
    plt.figure(figsize=(20,20))
    plt.imshow(rval,cmap='gray')
    for i in range(len(shapes)):
        shape = shapes[i];
        q = getShapeCoords(raster,shape, returnXY=True)
        plt.plot(q[0], q[1], color='red', alpha=0.7,
        linewidth=3, solid_capstyle='round', zorder=2)