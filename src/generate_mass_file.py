"""
BH masses can be incorrect in the BH merger catalogue.
This script sorts through the BH details text files and collects the masses of the BHs, by particle ID.
This is necessary for quick access when calculating the mass of the BH (by interpolation) at a specific scale factor.

The script uses MPI to parallelize the process of reading the files and collecting the data.

-----------------------------------------------------------------------------------------------

Stephanie Buttigieg (sb2583@cam.ac.uk)

"""

from mpi4py import MPI
import os
import numpy as np
import h5py

def process_file(file_path):
    masses = {}
    scale_factors = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = line.split()
            bh_id = int(data[0].split("=")[1])
            values = list(map(float, data[1:]))
            if bh_id in masses:
                masses[bh_id].append(values[1])
                scale_factors[bh_id].append(values[0])
            else:
                masses[bh_id] = [values[1]]
                scale_factors[bh_id] = [values[0]]
    return masses, scale_factors

def merge_dicts(dict_list):
    merged_masses = {}
    merged_scales = {}
    for masses, scales in dict_list:
        for bh_id in masses:
            if bh_id in merged_masses:
                merged_masses[bh_id].extend(masses[bh_id])
                merged_scales[bh_id].extend(scales[bh_id])
            else:
                merged_masses[bh_id] = masses[bh_id]
                merged_scales[bh_id] = scales[bh_id]
    return merged_masses, merged_scales

def main(verbose=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set the base path and output file as necessary
    basepath = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/blackhole_details'
    h5_output = '/cosma8/data/dp012/dc-butt3/bh_masses.hdf5'

    files = sorted(os.listdir(basepath))
    my_files = files[rank::size]

    local_data = []
    for file in my_files[:10]:
        path = os.path.join(basepath, file)
        if verbose:
            print(f'Rank {rank} processing file: {path}')
        local_data.append(process_file(path))

    # Gather data at rank 0
    gathered = comm.gather(local_data, root=0)

    if verbose:
        print('Gathered all data, now merging.')

    if rank == 0:
        flat_data = [item for sublist in gathered for item in sublist]
        masses, scale_factors = merge_dicts(flat_data)

        if verbose:
            print('Merged data, now writing to HDF5.')

        with h5py.File(h5_output, 'w') as f:
            for bh_id in masses:
                data = np.column_stack((scale_factors[bh_id], masses[bh_id]))
                f.create_dataset(str(bh_id), data=data)

        if verbose:
            print("Finished writing to HDF5 file.")

if __name__ == "__main__":
    main(verbose=True)
