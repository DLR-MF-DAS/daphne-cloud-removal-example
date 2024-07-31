import click
import rasterio
import numpy as np
import numpy.ma as ma
import pandas as pd
import glob
import os
import time
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
    input_files = list(glob.glob(os.path.join(input_dir, '*.tif')))
    for i in range(1, len(input_files)):
        tiff_data = []
        tiff_data_np = []
        distance_data = []
        mask_data = []
        profile = {}
        start_time = time.perf_counter()
        for tiff_file in input_files[:i]:
            with rasterio.open(tiff_file, 'r') as src:
                profile = src.profile
                data = src.read()
                tiff_data_np.append(data)
                data_shape = data.shape
                data_daphne = [dc.from_numpy(data[i].astype(np.int64)) for i in range(data.shape[0])]
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
                ds = [(band1 - band2) ** 2 for band1, band2 in zip(data1, data2)]
                d = ds[0]
                for d_ in ds[1:]:
                    d += d_
                d = d.sqrt()
                if distances is None:
                    distances = (d, new_mask)
                else:
                    distances = (distances[0] + d, daphne_and(distances[1], new_mask))
            distance_data.append(distances)
        distance_data_np = []
        for distance in distance_data:
            distance_data_np.append(ma.masked_array(distance[0].compute(), distance[1].compute()))
        ix = np.array([ma.argmin(distance_data_np, axis=0, keepdims=True)])
        composite = np.take_along_axis(np.array(tiff_data_np), ix, axis=0)[0]
        with rasterio.open(output_file + f"{i}.tif", 'w', **profile) as dst:
            dst.write(composite)
        end_time = time.perf_counter()
        print(end_time - start_time)
    

if __name__ == '__main__':
    main()
