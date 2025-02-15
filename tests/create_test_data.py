import pdal
import numpy as np
from math import sqrt, floor


# Create a pointcloud dataset that follows expected paramters for data testing
# purposes. Should be:
# - Uniform density
# - have intensional holes? Maybe should make a second for this
# - have expected attributes (XYZ)
# - have nice round bounds
# - be in CRS 5070

filename = './data/test_data_pixel_area.copc.laz'
cell_size = 30

# has a nice square root of 300
num_points = 90000
split = sqrt(num_points)
interval = 1

# making it square
minx = 300
maxx = minx + split
miny = minx
maxy = maxx
# to easily make datasets with consistently different values
diff_maker = 0

x_pos = np.arange(minx, maxx, interval, dtype=np.float32)
y_pos = np.arange(miny, maxy, interval, dtype=np.float32)
# positions = pos[np.where(pos % cell_size != 0)]

alg = lambda y: floor(y/cell_size) + diff_maker

data = np.array([(x, y, alg(y), alg(y), alg(y), alg(y))
                  for x in x_pos for y in y_pos],
    dtype=[
        ('X', np.float32),
        ('Y', np.float32),
        ('Z', np.float32),
        ('Intensity', np.uint16),
        ('NumberOfReturns', np.uint8),
        ('ReturnNumber', np.uint8)
    ]
)

print(f'writing out to {filename}')

p: pdal.Pipeline = pdal.Pipeline(arrays=[data]) | pdal.Writer(filename, a_srs='EPSG:5070', scale_x=0.01, scale_y=0.01, scale_z=0.01, forward='all')
p.execute()

# perform the same function, but adjust to a buffered size of +0.5*resolution
# to make pixel point tests have same results

filename = './data/test_data_pixel_point.copc.laz'
cell_size = 30

# making it square
minx -= cell_size / 2
maxx += cell_size / 2
miny -= cell_size / 2
maxy += cell_size / 2

# has a nice square root of 300
num_points = (maxx - minx) ** 2 # 108900.0
split = maxx - minx

interval = (maxx - minx) / split

# to easily make datasets with consistently different values
diff_maker = 0

x_pos = np.arange(minx, maxx, interval, dtype=np.float32)
y_pos = np.arange(miny, maxy, interval, dtype=np.float32)
# positions = pos[np.where(pos % cell_size != 0)]

alg = lambda y: floor((y+15)/cell_size) + diff_maker

data = np.array([(x, y, alg(y), alg(y), alg(y), alg(y))
                  for x in x_pos for y in y_pos],
    dtype=[
        ('X', np.float32),
        ('Y', np.float32),
        ('Z', np.float32),
        ('Intensity', np.uint16),
        ('NumberOfReturns', np.uint8),
        ('ReturnNumber', np.uint8)
    ]
)

print(f'writing out to {filename}')

p: pdal.Pipeline = pdal.Pipeline(arrays=[data]) | pdal.Writer(filename, a_srs='EPSG:5070', scale_x=0.01, scale_y=0.01, scale_z=0.01, forward='all')
p.execute()