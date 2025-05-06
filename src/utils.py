"""
Simple utility functions for the FABLE simulation.

----------------------------------------------------------------

Stephanie Buttigieg (sb2583@cam.ac.uk)

"""

import numpy as np
import h5py
import os

def get_scale_factors():
    """
    Read the scale factors from ExpansionList_128.

    Returns
    -------
    scale_factors : numpy.ndarray
        Array of scale factors.
    """
    path_to_expansion_list = '/cosma/home/dp012/dc-butt3/BHMacroDelay/data/required_data/expansion_list.txt' # modify this as required
    scale_factors = []
    with open(path_to_expansion_list, 'r') as f:
        for line in f:
            line_data = line.split()
            scale_factors.append(float(line_data[0]))
    scale_factors = np.array(scale_factors)
    return scale_factors

def get_subfind_ID(subhalo_index, snap):
    """
    Get the subfind ID of a subhalo at a given snapshot which is needed to identify subhalo in merger files.

    Parameters
    ----------
    subhalo_index : int
        Index of the subhalo in the merger tree.
    snap : int
        Snapshot number.
    Returns
    -------
    subfindID : int
        Subfind ID of the subhalo.
    """
    tree_path = '/cosma7/data/dp012/dc-butt3/MergerTrees/output/Galaxies/FABLE/Fid_test' # path to merger trees
    offsets_path = os.path.join(tree_path, 'offsets') # path to offsets of subhalo merger trees

    offsets_file_path = os.path.join(offsets_path, f'offsets_{snap:03}.hdf5')
    with h5py.File(offsets_file_path, 'r') as hf_offsets:
        subhaloIDs = np.array(hf_offsets['SubhaloID']) # this is the sublink ID, this is what I need to identify the subhalo and load further information.
        subhaloID = subhaloIDs[subhalo_index]

    if subhaloID == -1:
        subfindID = -1
    
    else:
        tree_number = int(np.floor(subhaloID/1e16))
        tree_path = os.path.join(tree_path, f'tree_extended.{tree_number}.hdf5')

        with h5py.File(tree_path, 'r') as hf:
            subhalo_indices = np.array(hf['SubhaloID'])
            index = np.where(subhalo_indices == subhaloID)[0][0]
            subfindID = np.array(hf['SubfindID'])[index]

    return subfindID