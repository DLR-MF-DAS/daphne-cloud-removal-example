import click
import rasterio
import numpy as np
import numpy.ma as ma
import glob
import os
from operator import and_
from api.python.context.daphne_context import DaphneContext


BAND_NAMES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'BQA10', 'BQA20', 'BQA60', 'FILL_MASK', 'CLOUD_MASK', 'CLOUDLESS_MASK', 'SHADOW_MASK', 'CLOUD_PROB', 'CLOUD_DIST']

BAND_DICT = dict([(band_name, i) for i, band_name in enumerate(BAND_NAMES)])

@click.command()
@click.option('-i', '--input-dir', help='Input directory', required=True)
@click.option('-o', '--output-file', help='Output filename', required=True)
def main(input_dir, output_file):
    dc = DaphneContext()
    tiff_data = []
    distance_data = []
    mask_data = []
    profile = {}
    for tiff_file in glob.glob(os.path.join(input_dir, '*.tif')):
        with rasterio.open(tiff_file, 'r') as src:
            profile = src.profile
            data = src.read()
            data_daphne = dc.from_numpy(data[0])
            mask = np.logical_or(data[BAND_DICT['CLOUD_MASK']], data[BAND_DICT['SHADOW_MASK']])
            mask = np.logical_or(mask, np.logical_not(data[BAND_DICT['FILL_MASK']]))
            mask = np.tile(mask, (22, 1, 1))
            masked_data = ma.masked_array(data, mask)
            tiff_data.append(masked_data)
    for tiff_file in tiff_data:
        distances = None
        for other_file in tiff_data:
            data1 = tiff_file[0:13]
            data2 = other_file[0:13]
            d = ma.sqrt(((data1 - data2) ** 2).sum(axis=0))
            if distances is None:
                distances = d
            else:
                distances = ma.masked_array(distances.data + d.data, mask=np.logical_and(distances.mask, d.mask))
        distance_data.append(distances)
    ix = np.array([ma.argmin(distance_data, axis=0, keepdims=True)])
    composite = np.take_along_axis(np.array(tiff_data), ix, axis=0)[0]
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(composite)
    

if __name__ == '__main__':
    main()
