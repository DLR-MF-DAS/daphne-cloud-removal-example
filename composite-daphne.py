import click
import rasterio
import numpy as np
import numpy.ma as ma
import pandas as pd
import glob
import os
from operator import and_
from daphne.context.daphne_context import DaphneContext


BAND_NAMES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'BQA10', 'BQA20', 'BQA60', 'FILL_MASK', 'CLOUD_MASK', 'CLOUDLESS_MASK', 'SHADOW_MASK', 'CLOUD_PROB', 'CLOUD_DIST']

BAND_DICT = dict([(band_name, i) for i, band_name in enumerate(BAND_NAMES)])

def daphne_not(arr):
    return (arr - 1) * (-1)

def daphne_and(arr1, arr2):
    return arr1 * arr2

def daphne_or(arr1, arr2):
    return daphne_not(daphne_not(arr1) * daphne_not(arr2))

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
            data_daphne = [dc.from_numpy(data[i].astype(np.int32)) for i in range(data.shape[0])]
            mask = daphne_or(data_daphne[BAND_DICT['CLOUD_MASK']], data_daphne[BAND_DICT['SHADOW_MASK']])
            mask = daphne_or(mask, daphne_not(data_daphne[BAND_DICT['FILL_MASK']]))
            tiff_data.append((data_daphne, mask))
    for tiff_file in tiff_data:
        distances = None
        for other_file in tiff_data:
            data1, mask1 = tiff_file
            data2, mask2 = other_file
            data1 = data1[0:13]
            data2 = data2[0:13]
            new_mask = daphne_and(mask1, mask2)
            ds = [(band1 - band2).sqrt() for band1, band2 in zip(data1, data2)]
            import pdb; pdb.set_trace()
            d = ds[0]
            for d_ in ds[1:]:
                d += d_
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
