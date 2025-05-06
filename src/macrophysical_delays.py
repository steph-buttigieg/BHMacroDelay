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
from scipy.interpolate import interp1d
import astropy.constants as const
import astropy.units as u
from scipy.spatial import cKDTree
import illustris_python.snapshot as snapshot
from scipy.integrate import solve_ivp

LITTLE_H = 0.679
DM_MASS = 3.4e7 / LITTLE_H
SOFTENING_LENGTH = 14.3606
STELLAR_MASS = 6.4e6/LITTLE_H

basepath = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100'
path_to_extended_merger_files = '/cosma7/data/dp012/dc-butt3/new_merger_events_extended' # change as required
treedir = '/cosma7/data/dp012/dc-butt3/MergerTrees/output/Galaxies/FABLE/Fid_test' # path to subhalo merger trees

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
    path_to_files = '../data/generated_data/delay_data/'

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


def get_masses_after_delay(merger_index, initial_m1, initial_m2, verbose = False):
    """
    Calculate the masses of the black holes after the macrophysical time delay.
    This is done by loading gas properties around the host subhalo position and calculating the accretion rate.
    The method outputs a detailed file for each merger event with the following format:

    BH_1 information:
    snap_number, mass, accretion_rate

    BH_2 information:
    snap_number, mass, accretion_rate

    If the accretion rate cannot be calculated, the accretion rate is set to -1.


    Parameters
    ----------
    merger_index : str
        The index of the merger event to process in the format 'id1_id2'.
    initial_m1 : float
        The initial mass of the first black hole in solar masses.
    initial_m2 : float
        The initial mass of the second black hole in solar masses.
    verbose : bool, optional
        If True, print progress messages. The default is False.
    Returns
    -------
    initial_m1 : float
        The mass of the first black hole after the time delay in solar masses.
    initial_m2 : float
        The mass of the second black hole after the time delay in solar masses.
    """
    path_to_detailed_output = '/cosma7/data/dp012/dc-butt3/accretion_details' # where detailed output files will be stored
    path_to_results = os.path.join(path_to_detailed_output, f'{merger_index}.txt')

    # check whether folder exists
    if not os.path.exists(path_to_detailed_output):
        os.makedirs(path_to_detailed_output)
    
    # load pressure threshold data to apply the same criteria as in the simulation
    pressure_file_path = os.path.join(basepath, 'bh_pressure_threshold.txt')
    data = np.loadtxt(pressure_file_path)
    m_bh = data[:,2]*1e10/LITTLE_H
    ref_press = data[:,3]
    log_m_bh = np.log10(m_bh)
    log_ref_press = np.log10(ref_press)
    interp_log_ref_press = interp1d(log_m_bh, log_ref_press, kind='linear', fill_value='extrapolate')

    def get_ref_press(m_bh):
        log_m_bh = np.log10(m_bh)
        log_ref_press = interp_log_ref_press(log_m_bh)
        return 10**log_ref_press
    
    # get macrophysical delay information
    galaxy_z, galaxy_merger_snap, _, _, initial_snap = process_one(merger_index)

    if galaxy_merger_snap == -1:
        # this happens when I have a time delay exactly equal to 0, so I don't need to calculate accretion.

        with open(path_to_results, 'w') as f:
            f.write(f'{galaxy_z} {initial_m1} -1 \n')
            f.write('\n')
            f.write(f'{galaxy_z} {initial_m2} -1 \n')

        return initial_m1, initial_m2
    
    # load information from detailed merger files (generated using generate_merger_files.py) 
    filename = f'{merger_index}.hdf5'
    filepath = os.path.join(path_to_extended_merger_files, filename)
    with h5py.File(filepath, 'r') as hf:
        header = dict(hf['Header'].attrs.items())
        snap_numbers_1 = np.array(hf['bh1']['SnapNum'])
        snap_numbers_2 = np.array(hf['bh2']['SnapNum'])
        group_number_1 = np.array(hf['bh1']['Group_Number'])
        group_number_2 = np.array(hf['bh2']['Group_Number'])
        snap_before_BH_merger = snap_numbers_1[0]
        subfind_id_1 = np.array(hf['bh1']['SubfindID'])
        subfind_id_2 = np.array(hf['bh2']['SubfindID'])
        BH_mass_1 = header['m1']
        BH_mass_2 = header['m2']
        time_of_numerical_merger = header['formation_scale_factor']
        z_numerical_merger = 1/time_of_numerical_merger -1

    if snap_before_BH_merger == galaxy_merger_snap: # don't have any data to calculate accretion.

        with open(path_to_results, 'w') as f:
            f.write(f'{galaxy_z} {initial_m1} -1 \n')
            f.write('\n')
            f.write(f'{galaxy_z} {initial_m2} -1 \n')

        return BH_mass_1, BH_mass_2
    
    if z_numerical_merger < galaxy_z:

        with open(path_to_results, 'w') as f:
            f.write(f'{galaxy_z} {initial_m1} -1 \n')
            f.write('\n')
            f.write(f'{galaxy_z} {initial_m2} -1 \n')
            
        return BH_mass_1, BH_mass_2
    
    subfind_id_1 = dict(zip(snap_numbers_1, subfind_id_1))
    subfind_id_2 = dict(zip(snap_numbers_2, subfind_id_2))
    group_number_1 = dict(zip(snap_numbers_1, group_number_1))
    group_number_2 = dict(zip(snap_numbers_2, group_number_2))

    initial_bh_mass_1 = initial_m1
    initial_bh_mass_2 = initial_m2

    if verbose:
        print('Initial BH mass 1 = ', initial_bh_mass_1)
        print('Initial BH mass 2 = ', initial_bh_mass_2)

    scale_factors = util.get_scale_factors()

    # load information about the host galaxies from the subhalo merger trees.
    tree = readtreeHDF5.TreeDB(treedir)
    subhalo_quants = ['SubhaloGrNr', 'SubhaloPos', 'SnapNum', 'SubfindID', 'SubhaloVmax']

    subfindID1 = subfind_id_1[initial_snap]
    subfindID2 = subfind_id_2[initial_snap]

    past_branch1 = tree.get_main_branch(initial_snap, subfindID1, keysel=subhalo_quants)
    fut_branch1 = tree.get_future_branch(initial_snap, subfindID1, keysel=subhalo_quants)

    past_branch2 = tree.get_main_branch(initial_snap, subfindID2, keysel=subhalo_quants)
    fut_branch2 = tree.get_future_branch(initial_snap, subfindID2, keysel=subhalo_quants)

    for b,branch in enumerate([past_branch1, fut_branch1]):
        snaps = branch.SnapNum
        host1_pos = branch.SubhaloPos 
        subfind_id1 = branch.SubfindID
        group_number1 = branch.SubhaloGrNr
        if b>0:
            pos_dict_future = dict(zip(snaps, host1_pos))
            subfind_dict_future1 = dict(zip(snaps, subfind_id1))
            group_number_dict_future1 = dict(zip(snaps, group_number1))
            pos_dict_combined1 = {**pos_dict_past, **pos_dict_future}
            subfind_dict_combined1 = {**subfind_dict_past1, **subfind_dict_future1}
            group_number_dict_combined1 = {**group_number_dict_past1, **group_number_dict_future1}
        else:
            pos_dict_past = dict(zip(snaps, host1_pos))
            subfind_dict_past1 = dict(zip(snaps, subfind_id1))
            group_number_dict_past1 = dict(zip(snaps, group_number1))

        for b,branch in enumerate([past_branch2, fut_branch2]):
            snaps = branch.SnapNum
            host2_pos = branch.SubhaloPos
            subfind_id2 = branch.SubfindID
            group_number2 = branch.SubhaloGrNr
            if b>0:
                pos_dict_future = dict(zip(snaps, host2_pos))
                subfind_dict_future2 = dict(zip(snaps, subfind_id2))
                group_number_dict_future2 = dict(zip(snaps, group_number2))
                pos_dict_combined2 = {**pos_dict_past, **pos_dict_future}
                subfind_dict_combined2 = {**subfind_dict_past2, **subfind_dict_future2}
                group_number_dict_combined2 = {**group_number_dict_past2, **group_number_dict_future2}
            else:
                pos_dict_past = dict(zip(snaps, host2_pos))
                subfind_dict_past2 = dict(zip(snaps, subfind_id2))
                group_number_dict_past2 = dict(zip(snaps, group_number2))

    def get_cs_T(internal_energy, e_abundance):
        """ Calculates the sound speed and temperature from the internal energy and electron abundance."""
        gamma = 5/3 # adiabatic index
        X_H = 0.76 # hydrogen mass fraction
        internal_energy = internal_energy * (u.km/u.s)**2
        mu = 4 * const.m_p/(1+3*X_H + 4*X_H*e_abundance) # mean molecular weight
        # convert internal energy to temperature
        T = internal_energy*(gamma-1)*mu/const.k_B
        # get sound speed
        c_s = np.sqrt(gamma*const.k_B*T/(mu))
        return c_s.to(u.km/u.s), T.to(u.K)

    def get_gas_data(snap, subhalo_pos, group_number):
        """ Loads gas data from the host halo around the host subhalo position and calculates the sound speed, density and pressure."""
        halo_data = snapshot.loadHalo(basepath, snap, group_number, partType = 0, fields = ['Coordinates', 'Masses', 'InternalEnergy', 'ElectronAbundance', 'Density'])
        if 'Coordinates' not in halo_data:
            # if there is no gas in the halo, return 0 for sound speed, density and pressure
            return 0, 0, 0
        gas_coordinates = halo_data['Coordinates']
        gas_masses = halo_data['Masses']
        gas_internal_energy = halo_data['InternalEnergy']
        gas_e_abundance = halo_data['ElectronAbundance']
        gas_density = halo_data['Density']
        number_of_neighbours = 32 # number of neighbours to use for the calculation of the sound speed
        if len(gas_coordinates) < number_of_neighbours:
            # in this case I just assume that the gas density is not high enough for the BH to accrete.
            return 0, 0, 0
        tree = cKDTree(gas_coordinates, boxsize = 100000)
        _, indices = tree.query(subhalo_pos, k=number_of_neighbours)
        chosen_gas_masses = gas_masses[indices]
        chosen_gas_internal_energy = gas_internal_energy[indices]
        chosen_gas_e_abundance = gas_e_abundance[indices]
        chosen_gas_density = gas_density[indices]
        c_s, _ = get_cs_T(chosen_gas_internal_energy, chosen_gas_e_abundance) # in km/s

        # calculate the mass-weighted average c_s
        c_s = np.sum(chosen_gas_masses*c_s)/np.sum(chosen_gas_masses)

        rho = np.sum(chosen_gas_masses*chosen_gas_density)/np.sum(chosen_gas_masses)

        gamma = 5/3
        average_u = np.sum(chosen_gas_masses*chosen_gas_internal_energy)/np.sum(chosen_gas_masses)
        pressure = (gamma-1)*rho*average_u

        return c_s.value, rho, pressure # rho is in code units
    
    def get_data_for_snap(snap, snap_before_BH_merger, galaxy_merger_snap, subfind_dict, group_dict, pos_dict, verbose=False):
        """Retrieve data for a given snap"""
        if snap in subfind_dict:
            new_snap = snap
        elif snap == snap_before_BH_merger:
            # Search backward until I find a valid snap
            new_snap = snap
            while new_snap not in subfind_dict:
                new_snap -= 1
        elif snap == galaxy_merger_snap:
            # Search forward
            new_snap = snap
            while new_snap not in subfind_dict and new_snap < 135:
                new_snap += 1
            if new_snap == 135 and new_snap not in subfind_dict:
                return 0, 0, 0, new_snap
        else:
            # if it is an 'internal' snap but is not in the dictionary we can skip this snap and then interpolate
            return None  

        group_number = group_dict[new_snap]
        host_pos = pos_dict[new_snap]
        c_s, rho, pressure = get_gas_data(new_snap, host_pos, group_number)
        return c_s, rho, pressure, new_snap

    if verbose:
        print('Last snap to load: ', galaxy_merger_snap)

    sound_speeds1 = []
    sound_speeds2 = []
    snaps1 = []
    snaps2 = []
    densities1 = []
    densities2 = []
    bh_pressure_1 = []
    bh_pressure_2 = []

    for snap in range(snap_before_BH_merger, galaxy_merger_snap + 1):
        if verbose:
            print('Loading snap', snap)

        # Subhalo 1
        result1 = get_data_for_snap(snap, snap_before_BH_merger, galaxy_merger_snap,
                                    subfind_dict_combined1, group_number_dict_combined1, pos_dict_combined1, verbose)
        if result1:
            c_s1, rho1, pressure1, used_snap1 = result1
            sound_speeds1.append(c_s1)
            densities1.append(rho1)
            bh_pressure_1.append(pressure1)
            snaps1.append(used_snap1)

        # Subhalo 2
        result2 = get_data_for_snap(snap, snap_before_BH_merger, galaxy_merger_snap,
                                    subfind_dict_combined2, group_number_dict_combined2, pos_dict_combined2, verbose)
        if result2:
            c_s2, rho2, pressure2, used_snap2 = result2
            sound_speeds2.append(c_s2)
            densities2.append(rho2)
            bh_pressure_2.append(pressure2)
            snaps2.append(used_snap2)

        if verbose and result1 and result2:
            print('c_s1:', c_s1, 'c_s2:', c_s2)
            print('rho1:', rho1, 'rho2:', rho2)
            print('pressure1:', pressure1, 'pressure2:', pressure2)

    sound_speeds1 = np.array(sound_speeds1)
    sound_speeds2 = np.array(sound_speeds2)
    no_gas_1 = np.all(sound_speeds1 == 0)
    no_gas_2 = np.all(sound_speeds2 == 0)

    def get_interpolated_value(t, quantity, times):
        """Interpolate the log of a quantity, fallback to linear if result is NaN
        Used to interpolate the sound speed, density and pressure between snapshots."""
        log_interp = interp1d(times, np.log10(quantity), kind='linear', fill_value='extrapolate')
        result = log_interp(t)
        if np.isnan(result):
            linear_interp = interp1d(times, quantity, kind='linear', fill_value='extrapolate')
            return linear_interp(t)
        return 10**result
    
    def rate_of_change(t, M_BH):
        """Compute BH mass accretion rate using Bondi or Eddington limit."""
        t *= u.Gyr
        M_BH *= u.Msun
        epsilon_r = 0.1

        density = get_interpolated_value(t, densities, time_at_snaps) * u.Msun / u.pc**3
        c_s = get_interpolated_value(t, sound_speeds, time_at_snaps) * u.km / u.s

        if c_s == 0 or density == 0:
            return [0]

        p_ext = get_interpolated_value(t, pressures, time_at_snaps)
        p_ref = get_ref_press(M_BH.value)
        pre_factor = (p_ext / p_ref)**2 if p_ext < p_ref else 1

        bondi = 4 * 100 * np.pi * const.G**2 * M_BH**2 * density / c_s**3 * pre_factor
        edd = 4 * np.pi * const.G * M_BH * const.m_p / (0.1 * const.sigma_T * const.c)

        acc_rate = min(bondi, edd)
        return acc_rate.to(u.Msun / u.Gyr).value * (1 - epsilon_r)
    
    def evolve_bh_mass(initial_mass, dens, cs, press, timesnaps):
        """Evolve BH mass over time and return masses and accretion rates at snapshots."""
        global densities, sound_speeds, pressures, time_at_snaps
        densities, sound_speeds, pressures, time_at_snaps = dens, cs, press, timesnaps

        t_span = [0, time_at_snaps[-1]]
        t_eval = np.clip(np.logspace(np.log10(t_span[0] + 1e-9), np.log10(t_span[1] - 1e-9), 1000), *t_span)

        solution = solve_ivp(rate_of_change, t_span=t_span, y0=[initial_mass], t_eval=t_eval)
        interp_mass = interp1d(solution.t, solution.y[0], kind='linear', bounds_error=False, fill_value="extrapolate")
        masses = interp_mass(time_at_snaps)
        rates = [rate_of_change(t, [m])[0] for t, m in zip(time_at_snaps, masses)]
        return masses, rates, solution.y[0][-1]

    # Cosmology setup
    H0 = LITTLE_H * 100
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3065)
    age_at_merger = cosmo.age(1 / time_of_numerical_merger - 1).value

    # Time at snapshots in relation to the numerical merger.
    def get_time_at_snaps(snaps):
        z = 1 / scale_factors[snaps] - 1
        return cosmo.age(z).value - age_at_merger

    time_at_snaps1 = get_time_at_snaps(snaps1)
    time_at_snaps2 = get_time_at_snaps(snaps2)

    # BH 1 evolution
    if not no_gas_1:
        masses_at_snaps1, rates_at_snaps1, final_m1 = evolve_bh_mass(initial_bh_mass_1, densities1, sound_speeds1, bh_pressure_1, time_at_snaps1)
    else:
        final_m1 = initial_bh_mass_1
        masses_at_snaps1 = [final_m1] * len(time_at_snaps1)
        rates_at_snaps1 = [0] * len(time_at_snaps1)

    # BH 2 evolution
    if not no_gas_2:
        masses_at_snaps2, rates_at_snaps2, final_m2 = evolve_bh_mass(initial_bh_mass_2, densities2, sound_speeds2, bh_pressure_2, time_at_snaps2)
    else:
        final_m2 = initial_bh_mass_2
        masses_at_snaps2 = [final_m2] * len(time_at_snaps2)
        rates_at_snaps2 = [0] * len(time_at_snaps2)

    if verbose:
        print('Final BH mass 1 =', final_m1)
        print('Final BH mass 2 =', final_m2)

    with open(path_to_results, 'w') as f:
        for i,snap in enumerate(snaps1):
            f.write(f'{snap} {masses_at_snaps1[i]} {rates_at_snaps1[i]} \n')
        f.write('\n')
        for i,snap in enumerate(snaps2):
            f.write(f'{snap} {masses_at_snaps2[i]} {rates_at_snaps2[i]} \n')

    return final_m1, final_m2

def calculate_accretion(verbose=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    already_processed_events = []

    delay_path = '../data/generated_data/delay_data'
    accretion_path = '/cosma7/data/dp012/dc-butt3/accretion_details_new'
    if rank == 0:
        events_to_process = []
        # get all the events that have already been processed
        files = os.listdir(accretion_path)
        for file in files:
            index = file.split('.')[0]
            already_processed_events.append(index)
        # get all the file names
        files = os.listdir(delay_path)
        for file in files:
            with open(os.path.join(delay_path, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.split()
                    index1 = int(data[0])
                    index2 = int(data[1])
                    m1 = float(data[2])
                    m2 = float(data[3])
                    bh_z = float(data[4])
                    galaxy_z = float(data[5])
                    merged = data[6]
                    host_separation = data[7]
                    merger_index = f'{index1}_{index2}'
                    if merger_index not in already_processed_events:
                        events_to_process.append([merger_index, m1, m2, bh_z, galaxy_z, merged])
        chunks = np.array_split(events_to_process, size)
    else:
        chunks = None

    local_events = comm.scatter(chunks, root=0)
    local_indices = []
    local_data_1 = []
    local_data_2 = []
    local_data_3 = []
    local_data_4 = []
    local_data_5 = []

    for event in local_events:
        local_indices.append(event[0])
        local_data_1.append(event[1])
        local_data_2.append(event[2])
        local_data_3.append(event[3])
        local_data_4.append(event[4])
        local_data_5.append(event[5])

    for i, index in enumerate(local_indices):
        if verbose:
            print('Rank {} Processing merger {}'.format(rank, index))
        try:
            initial_m1 = float(local_data_1[i])
            initial_m2 = float(local_data_2[i])
            print('Initial masses:', initial_m1, initial_m2)
            m1_acc, m2_acc = get_masses_after_delay(index, initial_m1, initial_m2, verbose=True)
        except Exception as e:
            print('Error processing merger:', index)
            print(e)


if __name__ == "__main__":
    calculate_accretion(verbose=False)