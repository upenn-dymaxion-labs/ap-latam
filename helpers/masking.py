from rasterio._features import _rasterize
from rasterio.dtypes import validate_dtype, can_cast_dtype, get_minimum_dtype
from rasterio.enums import MergeAlg
from rasterio.transform import IDENTITY, guard_transform
import numpy as np

def getBinaryMask(dataset, shapes, transformed = True, window = None):
    """
    Returns a binary mask with respect to the shape polygons overlayed on the raster file
    
    Arguments : 
    dataset : Raster file object
    shapes : List of Dictionaries, with each dict containing geospatial coordinates of polygon vertices
    window : A window object specifying the specific window in the raster to be concentrating on. The fina;
             binary mask is cropped based on the given window

    """
    
    
    mask = rasterize(shapes,out_shape=(int(dataset.height), int(dataset.width)),
        transform=dataset.transform,
        all_touched=False,fill=1,default_value=0)
    
    if window!=None:
        windowed_mask = mask[window.row_off:(window.row_off+window.height) , window.col_off:(window.col_off+window.width)]
        return(windowed_mask)
    else:
        return(mask)
    
    return(mask)

def rasterize(
        shapes,
        out_shape=None,
        fill=0,
        out=None,
        transform=IDENTITY,
        all_touched=False,
        merge_alg=MergeAlg.replace,
        default_value=1,
        dtype=None):
    
    valid_dtypes = (
        'int16', 'int32', 'uint8', 'uint16', 'uint32', 'float32', 'float64'
    )

    def format_invalid_dtype(param):
        return '{0} dtype must be one of: {1}'.format(
            param, ', '.join(valid_dtypes)
        )

    def format_cast_error(param, dtype):
        return '{0} cannot be cast to specified dtype: {1}'.format(param, dtype)

    if fill != 0:
        fill_array = np.array([fill])
        if not validate_dtype(fill_array, valid_dtypes):
            raise ValueError(format_invalid_dtype('fill'))

        if dtype is not None and not can_cast_dtype(fill_array, dtype):
            raise ValueError(format_cast_error('fill', dtype))

    if default_value != 1:
        default_value_array = np.array([default_value])
        if not validate_dtype(default_value_array, valid_dtypes):
            raise ValueError(format_invalid_dtype('default_value'))

        if dtype is not None and not can_cast_dtype(default_value_array, dtype):
            raise ValueError(format_cast_error('default_vaue', dtype))

    if dtype is not None and np.dtype(dtype).name not in valid_dtypes:
        raise ValueError(format_invalid_dtype('dtype'))

    valid_shapes = []
    shape_values = []
    for index, item in enumerate(shapes):
        if isinstance(item, (tuple, list)):
            geom, value = item
        else:
            geom = item
            value = default_value
        geom = getattr(geom, '__geo_interface__', None) or geom

        # geom must be a valid GeoJSON geometry type and non-empty
        if not is_valid_geom(geom):
            raise ValueError(
                'Invalid geometry object at index {0}'.format(index)
            )

        if geom['type'] == 'GeometryCollection':
            # GeometryCollections need to be handled as individual parts to
            # avoid holes in output:
            # https://github.com/mapbox/rasterio/issues/1253.
            # Only 1-level deep since GeoJSON spec discourages nested
            # GeometryCollections
            for part in geom['geometries']:
                valid_shapes.append((part, value))

        else:
            valid_shapes.append((geom, value))

        shape_values.append(value)

    if not valid_shapes:
        raise ValueError('No valid geometry objects found for rasterize')

    shape_values = np.array(shape_values)

    if not validate_dtype(shape_values, valid_dtypes):
        raise ValueError(format_invalid_dtype('shape values'))

    if dtype is None:
        dtype = get_minimum_dtype(np.append(shape_values, fill))

    elif not can_cast_dtype(shape_values, dtype):
        raise ValueError(format_cast_error('shape values', dtype))

    if out is not None:
        if np.dtype(out.dtype).name not in valid_dtypes:
            raise ValueError(format_invalid_dtype('out'))

        if not can_cast_dtype(shape_values, out.dtype):
            raise ValueError(format_cast_error('shape values', out.dtype.name))

    elif out_shape is not None:

        if len(out_shape) != 2:
            raise ValueError('Invalid out_shape, must be 2D')

        out = np.empty(out_shape, dtype=dtype)
        out.fill(fill)

    else:
        raise ValueError('Either an out_shape or image must be provided')

    if min(out.shape) == 0:
        raise ValueError("width and height must be > 0")

    transform = guard_transform(transform)
    _rasterize(valid_shapes, out, transform, all_touched, merge_alg)
    return out

def createMask(x, threshold):    
    y = x.copy()
    y[y < threshold] = 0.0
    return(y)

def is_valid_geom(geom):
    """
    Checks to see if geometry is a valid GeoJSON geometry type or
    GeometryCollection.

    Geometries must be non-empty, and have at least x, y coordinates.

    Note: only the first coordinate is checked for validity.

    Parameters
    ----------
    geom: an object that implements the geo interface or GeoJSON-like object

    Returns
    -------
    bool: True if object is a valid GeoJSON geometry type
    """

    geom_types = {'Point', 'MultiPoint', 'LineString', 'LinearRing',
                  'MultiLineString', 'Polygon', 'MultiPolygon'}

    if 'type' not in geom:
        return False

    try:
        geom_type = geom['type']
        if geom_type not in geom_types.union({'GeometryCollection'}):
            return False

    except TypeError:
        return False

    if geom_type in geom_types:
        if 'coordinates' not in geom:
            return False

        coords = geom['coordinates']

        if geom_type == 'Point':
            # Points must have at least x, y
            return len(coords) >= 2

        if geom_type == 'MultiPoint':
            # Multi points must have at least one point with at least x, y
            return len(coords) > 0 and len(coords[0]) >= 2

        if geom_type == 'LineString':
            # Lines must have at least 2 coordinates and at least x, y for
            # a coordinate
            return len(coords) >= 2 and len(coords[0]) >= 2

        if geom_type == 'LinearRing':
            # Rings must have at least 4 coordinates and at least x, y for
            # a coordinate
            return len(coords) >= 4 and len(coords[0]) >= 2

        if geom_type == 'MultiLineString':
            # Multi lines must have at least one LineString
            return (len(coords) > 0 and len(coords[0]) >= 2 and
                    len(coords[0][0]) >= 2)

        if geom_type == 'Polygon':
            # Polygons must have at least 1 ring, with at least 4 coordinates,
            # with at least x, y for a coordinate
            return (len(coords) > 0 and len(coords[0]) >= 4 and
                    len(coords[0][0]) >= 2)

        if geom_type == 'MultiPolygon':
            # Muti polygons must have at least one Polygon
            return (len(coords) > 0 and len(coords[0]) > 0 and
                    len(coords[0][0]) >= 4 and len(coords[0][0][0]) >= 2)

    if geom_type == 'GeometryCollection':
        if 'geometries' not in geom:
            return False

        if not len(geom['geometries']) > 0:
            # While technically valid according to GeoJSON spec, an empty
            # GeometryCollection will cause issues if used in rasterio
            return False

        for g in geom['geometries']:
            if not is_valid_geom(g):
                return False  # short-circuit and fail early

    return True