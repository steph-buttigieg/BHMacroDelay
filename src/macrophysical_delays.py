"""
This script calculates the macrophysical time delays for all merger events passing the mass criteria.
Note that you require the extended merger files to run this code, which can be generated using generate_merger_files.py.
This script will also include functionality to calculate the masses of the BHs after this added time delay in future versions.

----------------------------------------------------------------------------------------------------------------------------

Stephanie Buttigieg (sb2583@cam.ac.uk)
"""

import h5py
import numpy as np
import readtreeHDF5
from astropy.cosmology import FlatLambdaCDM
import os
import utils as util
from mpi4py import MPI

LITTLE_H = 0.679
DM_MASS = 3.4e7 / LITTLE_H
SOFTENING_LENGTH = 14.3606
STELLAR_MASS = 6.4e6/LITTLE_H

basepath = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100'
path_to_extended_merger_files = '/cosma7/data/dp012/dc-butt3/new_merger_events_extended' # change as required

def periodic_distance(pos1, pos2, box_size=1e5):
    """
    Calculate the periodic distance between two points in a periodic box.
    Parameters
    ----------
    pos1 : array_like
        First position vector.
    pos2 : array_like
        Second position vector.
    box_size : float, optional
        Size of the periodic box in code units. The default is 1e5.
    Returns
    -------
    float
        Periodic distance between pos1 and pos2 in code units
    """

    # Compute displacement
    displacement = pos2 - pos1
    
    # Apply periodic boundary conditions
    displacement = np.where(displacement > 0.5 * box_size, displacement - box_size, displacement)
    displacement = np.where(displacement < -0.5 * box_size, displacement + box_size, displacement)
    
    # Calculate the Euclidean distance
    distance = np.linalg.norm(displacement)
    
    return distance

def process_one(merger_index, verbose = False):
    """
    Process a single merger event and calculate the macrophysical time delay.
    Parameters
    ----------
    merger_index : str
        The index of the merger event to process in the format 'id1_id2'.
    verbose : bool, optional
        If True, print progress messages. The default is False.
    Returns
    -------
    galaxy_z : float
        The redshift at which the host galaxies merged.
    host_merger_snap : int
        The snapshot number at which the host galaxies merged.
    merge_by_now : bool
        Whether the merger happened before redshift 0.
    distance : float
        The distance between the two galaxies at the snapshot before their merger.
    initial_snap : int
        The snapshot number at which the host subhalose are identified.
    """
    filename = f'{merger_index}.hdf5'

    def load_bh_data(hf, bh_key):
        bh_data = hf[bh_key]
        snap_numbers = np.array(bh_data['SnapNum'])
        return {
            'Group_Number': dict(zip(snap_numbers, np.array(bh_data['Group_Number']))),
            'Group_R_Crit200': dict(zip(snap_numbers, np.array(bh_data['Group_R_Crit200']))),
            'BH_Mass': dict(zip(snap_numbers, np.array(bh_data['BH_Mass']))),
            'Coordinates': dict(zip(snap_numbers, np.array(bh_data['Coordinates']))),
            'SubfindID': dict(zip(snap_numbers, np.array(bh_data['SubfindID']))),
            'SubhaloMassInRad': dict(zip(snap_numbers, np.array(bh_data['SubhaloMassInRad']))),
            'Central_Subhalo': dict(zip(snap_numbers, np.array(bh_data['Central_Subhalo']))),
        }, snap_numbers

    file_path = os.path.join(path_to_extended_merger_files, filename)

    with h5py.File(file_path, 'r') as hf:
        header = dict(hf['Header'].attrs.items())
        id1, id2 = header['id1'], header['id2']
        formation_time = header['formation_scale_factor']

        bh1_data, snap_numbers = load_bh_data(hf, 'bh1')
        bh2_data, _ = load_bh_data(hf, 'bh2')

    merger_snap = snap_numbers[0]

    # What are the common snap numbers where both BHs exist
    common_keys = set(bh1_data['BH_Mass'].keys()).intersection(set(bh2_data['BH_Mass'].keys()))
    # Findin the snaps where both BHs are in central subhalos
    filtered_keys = [key for key in common_keys if bh1_data['Central_Subhalo'][key] == 1 and bh2_data['Central_Subhalo'][key] == 1 and bh1_data['Group_Number'][key] != bh2_data['Group_Number'][key]]

    if filtered_keys:
        initial_snap = max(filtered_keys)
        if verbose:
            print('Largest snap where both BHs were in central subhalos:', initial_snap)
    else:
        initial_snap = min(common_keys) 
        if verbose:
            print('No common snap found so using the earliest snap where both BHs exist: ', initial_snap)

    total_mass_1 = bh1_data['SubhaloMassInRad'][initial_snap]*1e10/LITTLE_H
    total_mass_2 = bh2_data['SubhaloMassInRad'][initial_snap]*1e10/LITTLE_H
    snap_to_add_1 = 1

    # Check snap after if the mass is negative (i.e. the BH is not in a subhalo)
    while total_mass_1 < 0:
        try:
            total_mass_1 = bh1_data['Central_Subhalo'][initial_snap + snap_to_add_1]*1e10/LITTLE_H
        except KeyError:
            total_mass_1 = 0
        snap_to_add_1 += 1
    snap_to_add_2 = 1
    while total_mass_2 < 0:
        try:
            total_mass_2 =  bh2_data['Central_Subhalo'][initial_snap + snap_to_add_2]*1e10/LITTLE_H
        except KeyError:
            total_mass_2 = 0
        snap_to_add_2 += 1
    
    # Only keep processing event if the host subhalos at initial_snap are massive enough.
    if total_mass_1 < 100*STELLAR_MASS and total_mass_2 < 100*STELLAR_MASS:
        return None, None, None, None, None
    
    treedir = '/cosma7/data/dp012/dc-butt3/MergerTrees/output/Galaxies/FABLE/Fid_test' # path to subhalo merger trees
    tree = readtreeHDF5.TreeDB(treedir)

    # Ensuring that both host subhalos have a valid subfindID
    good_snap = False

    while not good_snap:
        if bh1_data['SubfindID'][initial_snap] != -1 and bh2_data['SubfindID'][initial_snap] != -1:
            good_snap = True
        if not good_snap: # move forward in time until both subfindIDs are valid.
            initial_snap += 1

    # Load subhalo data from the merger trees

    subhalo_quants = ['SubhaloGrNr', 'SubhaloPos', 'SnapNum', 'SubfindID', 'SubhaloVmax']

    subfindID1 = bh1_data['SubfindID'][initial_snap]
    subfindID2 = bh2_data['SubfindID'][initial_snap]

    past_branch1 = tree.get_main_branch(initial_snap, subfindID1, keysel=subhalo_quants)
    fut_branch1 = tree.get_future_branch(initial_snap, subfindID1, keysel=subhalo_quants)

    past_branch2 = tree.get_main_branch(initial_snap, subfindID2, keysel=subhalo_quants)
    fut_branch2 = tree.get_future_branch(initial_snap, subfindID2, keysel=subhalo_quants)

    for b,branch in enumerate([past_branch1, fut_branch1]):
        snaps = branch.SnapNum
        host1_pos = branch.SubhaloPos 
        subfind_id1 = branch.SubfindID
        if b>0:
            pos_dict_future = dict(zip(snaps, host1_pos))
            subfind_dict_future1 = dict(zip(snaps, subfind_id1))
            pos_dict_combined1 = {**pos_dict_past, **pos_dict_future}
            subfind_dict_combined1 = {**subfind_dict_past1, **subfind_dict_future1}
        else:
            pos_dict_past = dict(zip(snaps, host1_pos))
            subfind_dict_past1 = dict(zip(snaps, subfind_id1))

        for b,branch in enumerate([past_branch2, fut_branch2]):
            snaps = branch.SnapNum
            host2_pos = branch.SubhaloPos
            subfind_id2 = branch.SubfindID
            if b>0:
                pos_dict_future = dict(zip(snaps, host2_pos))
                subfind_dict_future2 = dict(zip(snaps, subfind_id2))
                pos_dict_combined2 = {**pos_dict_past, **pos_dict_future}
                subfind_dict_combined2 = {**subfind_dict_past2, **subfind_dict_future2}
            else:
                pos_dict_past = dict(zip(snaps, host2_pos))
                subfind_dict_past2 = dict(zip(snaps, subfind_id2))

    # check when the two subhalos merge, if at all
    if initial_snap:
        merge_by_now = False
        merger_snap = snap_numbers[0]
        max_snap = min(max(subfind_dict_combined1.keys()), max(subfind_dict_combined2.keys()))
        host_merger_snap = 0
        scale_factors = util.get_scale_factors()
        for snap in range(merger_snap, max_snap):
            # two subhalos are considered merged when their subfindIDs are the same
            if snap in subfind_dict_combined1.keys() and snap in subfind_dict_combined2.keys():
                if subfind_dict_combined1[snap] == subfind_dict_combined2[snap]:
                    host_merger_snap = snap
                    merge_by_now = True
                    break

        if max_snap == 135 and host_merger_snap == 0: # this is the case where the host subhalos do not merge before redshift 0
            if verbose:
                print('The last snap before BH merger:', merger_snap)
                print('The host subhalos did not merge before redshift 0')
            host_merger_snap = max_snap
        elif max_snap !=135 and host_merger_snap == 0:
            # subhalos haven't merged before one of them ceases to exist.
            # in this case I will set the host_merger_snap to the last snap before the subhalos cease to exist
            # this could happen because of ex. tidal disruption events.
            if verbose:
                print('The last snap before BH merger:', merger_snap)
                print('The host subhalos do not merge before redshift 0')
            host_merger_snap = max(max(subfind_dict_combined1.keys()), max(subfind_dict_combined2.keys()))
        else:
            if verbose:
                print('The host subhalos merged at snap:', host_merger_snap)
                print('The last snap before BH merger:', merger_snap)
       
        if host_merger_snap != merger_snap:
            # calculate the time delay
            H0 = LITTLE_H * 100
            Om0 = 0.3065
            cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
            galaxy_merger_a = scale_factors[host_merger_snap]
            bh_z = 1/formation_time - 1
            galaxy_z = 1/galaxy_merger_a - 1
            age_bh = cosmo.age(bh_z)
            age_galaxy = cosmo.age(galaxy_z)
            time_delay = age_galaxy - age_bh
            if verbose:
                print('Time delay:', time_delay)
            # calculate the distance between the two galaxies at the snap before their merger
            if merge_by_now == True:
                i = 1
                while host_merger_snap-i not in pos_dict_combined1.keys() or host_merger_snap-i not in pos_dict_combined2.keys():
                    i += 1
                pos1 = pos_dict_combined1[host_merger_snap-i]
                pos2 = pos_dict_combined2[host_merger_snap-i]
            else:
                i = 0
                while host_merger_snap-i not in pos_dict_combined1.keys() or host_merger_snap-i not in pos_dict_combined2.keys():
                    i += 1
                pos1 = pos_dict_combined1[host_merger_snap-i]
                pos2 = pos_dict_combined2[host_merger_snap-i]
            distance = periodic_distance(pos1, pos2)*scale_factors[host_merger_snap-1]  / LITTLE_H #in kpc

        else: # the case where the host subhalos merge at the same snap as the BHs/before the BHs
            if merge_by_now == True:
                # get the latest positions of the host subhalos according to what keys are available
                keys1 = pos_dict_combined1.keys()
                keys2 = pos_dict_combined2.keys()
                # get the largest snap smaller than host_merger_snap that is available in both dictionaries
                keys1 = [key for key in keys1 if key < host_merger_snap]
                keys2 = [key for key in keys2 if key < host_merger_snap]
                if keys1 and keys2:
                    common_keys = set(keys1).intersection(set(keys2))
                    if common_keys:
                        initial_snap = max(common_keys)
                        pos1 = pos_dict_combined1[initial_snap]
                        pos2 = pos_dict_combined2[initial_snap]
                        distance = periodic_distance(pos1, pos2)*scale_factors[initial_snap] / LITTLE_H
                    else:
                        distance = -1
                else:
                    distance = -1
            else:
                pos1 = pos_dict_combined1[host_merger_snap]
                pos2 = pos_dict_combined2[host_merger_snap]
                distance = periodic_distance(pos1, pos2)*scale_factors[host_merger_snap] / LITTLE_H
            galaxy_z = 1/formation_time -1 # setting equal to the formation redshift of the BHs

    return galaxy_z, host_merger_snap, merge_by_now, distance, initial_snap

def get_delays_only(verbose = False):
    """
    Get the macrophysical time delays for all merger events in the directory
    This method does not calculate accretion onto BHs during macrophysical delays

    It generates text files with the following format:
    id1, id2, m1, m2, formation_z, galaxy_merger_z, merge_by_now, host_distance
    
    where id1 and id2 are the IDs of the two black holes, m1 and m2 are their masses,
    formation_z is the redshift at which the merger event happens in the simulation, galaxy_merger_z is the redshift at which the host galaxies merged,
    merge_by_now is a boolean indicating whether the merger happened before redshift 0, and host_distance is the distance between the two galaxies in the snapshot before their merger.

    Parameters
    ----------
    verbose : bool, optional
        If True, print progress messages. The default is False.
    Returns
    -------
    None
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mergers_path = '/cosma7/data/dp012/dc-butt3/new_merger_events_extended' # change as required
    path_to_files = '../data/new_delay_data/'

    if rank == 0:
        indices_processed = []
        # Check if the directory exists and if not create it
        if not os.path.exists(path_to_files):
            os.makedirs(path_to_files)
        # open all the files in this directory and add indices to indices_processed
        files = os.listdir(path_to_files)
        for file in files:
            if verbose:
                print('Processing file:', file)
            with open(os.path.join(path_to_files, file)) as f:
                lines = f.readlines()
                for line in lines:
                    data = line.split()
                    id1 = data[0]
                    id2 = data[1]
                    indices_processed.append(f'{id1}_{id2}')

        # Get the list of merger files to process
        merger_files = os.listdir(mergers_path)
        merger_files = [file for file in merger_files if file.endswith('.hdf5')]
        indices_to_process = []
        for filename in merger_files:
            index = filename.split('.')[0]
            
            if index not in indices_processed:
                indices_to_process.append(index)

        chunks = np.array_split(indices_to_process, size)
    else:
        chunks = None

    local_indices = comm.scatter(chunks, root=0)
    for index in local_indices:
        if verbose:
            print(f'Rank {rank} Processing merger:', index)
        with h5py.File(os.path.join(mergers_path, f'{index}.hdf5'), 'r') as hf:
            header = dict(hf['Header'].attrs.items())
            try:
                id1 = header['id1']
                id2 = header['id2']
                m1 = header['m1']
                m2 = header['m2']
                formation_z = header['formation_redshift']
            except KeyError:
                print('KeyError: id1 not found in header for merger:', index)
                continue
            
        try:
            galaxy_merger_redshift,_, merge_by_now, host_distance, _ = process_one(index)
        except Exception as e:
            print('Error processing merger:', index)
            print(e)
            continue
        
        if galaxy_merger_redshift is not None: # ignoring the ones which are not in well-resolved subhalos at the point of identification.
            with open(os.path.join(path_to_files, f'delays_{rank}.txt'), 'a') as f:
                f.write(str(id1) + ' ' + str(id2) + ' ' + str(m1) + ' ' + str(m2) + ' ' + str(formation_z) +' ' + str(galaxy_merger_redshift) + ' ' + str(merge_by_now) + ' ' + str(host_distance) + '\n')


if __name__ == "__main__":
    get_delays_only(verbose=True)