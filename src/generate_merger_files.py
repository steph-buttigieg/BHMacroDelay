"""
This script generates an hdf5 file for each BH merger event (passing the selection criteria).
It stores information related to the BH particle and its host subhalo/group at each snap before the merger.

It uses MPI to parallelize the process of reading the merger events and creating the hdf5 files.

------------------------------------------------------------------------------------------------
Stephanie Buttigieg (sb2583@cam.ac.uk)
"""

import h5py
import numpy as np
import os
from mpi4py import MPI
import utils as util
import bh_mergers as bh
import illustris_python.groupcat as gc
import illustris_python.snapshot as snapshot
import astropy.units as u

basepath = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100'

def create_file(merger_event, verbose = False):
    """
    Create an hdf5 file for a merger event.
    The file contains information about the black holes and their host subhalos/groups at each snapshot before the merger.
    Parameters
    ----------
    merger_event : bh.MergerEvent
        The merger event object containing the details of the merger.
    verbose : bool, optional
        If True, print progress messages. The default is False.
    Returns
    -------
    None.
    """
    filename = f'{merger_event.id1}_{merger_event.id2}'
    if verbose:
        print(f"Creating tree for merger event {filename}")
    output_dir = '/cosma7/data/dp012/dc-butt3/new_merger_events_extended' # modify this as required
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hdf5_filename = os.path.join(output_dir, f"{filename}.hdf5")

    snap_before_merger = merger_event.snap_no

    # Initialize data structures

    bh_data = {
        1:{"SnapNum": [], "SubhaloMassInRad": [], "SubhaloMassInHalfRadType": [], "SubhaloPos": [], "SubfindID": [],
           "Coordinates": [], "Group_R_Crit200": [], "Group_Number": [], "BH_Mass": [], "Central_Subhalo": []},
        2:{"SnapNum": [], "SubhaloMassInRad": [], "SubhaloMassInHalfRadType": [], "SubhaloPos": [], "SubfindID": [],
           "Coordinates": [], "Group_R_Crit200": [], "Group_Number": [], "BH_Mass": [], "Central_Subhalo": []},
    }

    # Initialize subhalo data, change fields as needed
    subhalo_quants = ["SubhaloMassInRad", "SubhaloMassInHalfRadType", "SubhaloPos"]

    def update_dictionaries(snap, look_for, gr_nr, subhalo_nr):
        which_host = []
        if look_for[0]:
            which_host.append(1)
        if look_for[1]:
            which_host.append(2)
        subhalo_data = gc.loadSubhalos(basepath, snap, fields=subhalo_quants)
        group_data = gc.loadHalos(basepath, snap, fields=['Group_R_Crit200', 'GroupFirstSub'])
        group_vir = group_data['Group_R_Crit200']
        group_first_sub = group_data['GroupFirstSub']
        bh_info = snapshot.loadSubset(basepath, snap, partType=5, fields=['Coordinates', 'BH_Mass', 'ParticleIDs'])
        for host in which_host:
            bh_id = merger_event.id1 if host == 1 else merger_event.id2
            bh_index = np.where(bh_info['ParticleIDs'] == bh_id)[0][0]
            bh_data[host]["SnapNum"].append(snap)
            bh_data[host]["BH_Mass"].append(bh_info['BH_Mass'][bh_index])
            bh_data[host]["Coordinates"].append(bh_info['Coordinates'][bh_index])
            if subhalo_nr[host-1] != -1:
                # black hole is in a subhalo and is the simplest case
                for key in subhalo_quants:
                    bh_data[host][key].append(subhalo_data[key][subhalo_nr[host-1]])
                bh_data[host]["Group_Number"].append(gr_nr[host-1])
                bh_data[host]["Group_R_Crit200"].append(group_vir[gr_nr[host-1]])
                if group_first_sub[gr_nr[host-1]] == subhalo_nr[host-1]:
                    bh_data[host]["Central_Subhalo"].append(1)
                else:
                    bh_data[host]["Central_Subhalo"].append(0)
                # find the subfindID of the subhalo
                subhaloID = util.get_subfind_ID(subhalo_nr[host-1], snap)
                bh_data[host]["SubfindID"].append(subhaloID)
            elif subhalo_nr[host-1] == -1 and gr_nr[host-1] != -1:
                # black hole is in a group but not in a subhalo
                bh_data[host]["SubfindID"].append(-1)
                bh_data[host]["Group_Number"].append(gr_nr[host-1])
                bh_data[host]["Group_R_Crit200"].append(group_vir[gr_nr[host-1]])
                bh_data[host]["SubhaloMassInRad"].append(-1)
                bh_data[host]["SubhaloMassInHalfRadType"].append([-1] * 6)
                bh_data[host]["SubhaloPos"].append([-1, -1, -1])
                bh_data[host]["Central_Subhalo"].append(0)
            else:
                # black hole is not in a group or subhalo
                bh_data[host]["SubfindID"].append(-1)
                bh_data[host]["Group_Number"].append(-1)
                bh_data[host]["Group_R_Crit200"].append(-1)
                bh_data[host]["SubhaloMassInRad"].append(-1)
                bh_data[host]["SubhaloMassInHalfRadType"].append([-1] * 6)
                bh_data[host]["SubhaloPos"].append([-1, -1, -1])
                bh_data[host]["Central_Subhalo"].append(0)

    snap_before_merger = merger_event.snap_no

    look_for_1 = True
    look_for_2 = True

    snap_no = snap_before_merger

    while look_for_1 or look_for_2:
        # if verbose:
        #     print(f"Looking for black holes in snapshot {snap_no} for merger event {filename}")
        
        gr_nr1, subhalo_nr1 = bh.get_bh_group_subhalo_nr(merger_event.id1, snap_no)
        gr_nr2, subhalo_nr2 = bh.get_bh_group_subhalo_nr(merger_event.id2, snap_no)

        if gr_nr1 is None:
            look_for_1 = False
        if gr_nr2 is None:
            look_for_2 = False

        if look_for_1 or look_for_2:
            update_dictionaries(snap_no, (look_for_1, look_for_2), (gr_nr1, gr_nr2), (subhalo_nr1, subhalo_nr2))
            snap_no -= 1

    # Write data to HDF5 file
    with h5py.File(hdf5_filename, "w") as f:
        header = f.create_group("Header")
        header.attrs.update({
            "id1": merger_event.id1,
            "id2": merger_event.id2,
            "formation_redshift": merger_event.formation_redshift,
            "m1": merger_event.mass1,
            "m2": merger_event.mass2,
            "formation_scale_factor": merger_event.formation_scale_factor,
            "binary_sep": merger_event.binary_separation.to(u.pc).value,
        })

        for which_host in [1, 2]:
            bh_group = f.create_group(f"bh{which_host}")
            for key, value in bh_data[which_host].items():
                bh_group.create_dataset(key, data=value)

    if verbose:
        print(f"Finished creating the tree for merger event {hdf5_filename}")

def generate_all_files(verbose=False):
    """
    Iterates through the raw bh merger files.
    It then corrects for the masses of the black holes at the time of merger.
    It then checks whether it passes the minimum mass criterion.
    If it does, it creates a merger event object and stores it in a list.
    The list is then split into chunks and sent to the different ranks.
    Each worker then creates a merger detail file for each merger event using the create_file() method.

    Parameters
    ----------
    verbose : bool, optional
        If True, print progress messages. The default is False.
    Returns
    -------
    None.
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load the list of files to process (done only on rank 0)
    if rank == 0:
        path_to_files = '/cosma7/data/dp012/dc-butt3/new_merger_events_extended'
        processed_files = [f for f in os.listdir(path_to_files) if os.path.isfile(os.path.join(path_to_files, f))]
        processed_mergers = set(f.split('.')[0] for f in processed_files)

        folder_path = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/blackhole_mergers'
        all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
        file_chunks = np.array_split(all_files, size)
    else:
        folder_path = None
        file_chunks = None
        processed_mergers = None

    # Broadcast folder path and processed mergers to all ranks
    folder_path = comm.bcast(folder_path, root=0)
    processed_mergers = comm.bcast(processed_mergers, root=0)

    # Scatter file chunks
    local_files = comm.scatter(file_chunks, root=0)

    # Each rank loads and filters merger events from its files
    local_events = []
    for filename in local_files:
        file_path = os.path.join(folder_path, filename)
        if verbose:
            print(f'[Rank {rank}] Reading {file_path}')
        with open(file_path, 'r') as f:
            for line in f:
                values = line.split()
                merger_event = bh.MergerEvent(values[1], values[2], values[4])
                if merger_event.mass1 >= 1e6 and merger_event.mass2 >= 1e6:
                    local_events.append(merger_event)

    # Gather all merger events at rank 0
    all_events = comm.gather(local_events, root=0)

    # On rank 0: flatten the list and redistribute
    if rank == 0:
        all_merger_events = [event for sublist in all_events for event in sublist]
        if verbose:
            print(f"[Rank 0] Total events gathered: {len(all_merger_events)}")
        chunks = np.array_split(all_merger_events, size)
        scale_factors = util.get_scale_factors()
    else:
        chunks = None
        scale_factors = None

    scale_factors = comm.bcast(scale_factors, root = 0)
    merger_events = comm.scatter(chunks, root = 0)
    processed_mergers = comm.bcast(processed_mergers, root = 0)

    for event in merger_events:
        if f'{event.id1}_{event.id2}' not in processed_mergers:
            create_file(event, verbose=verbose)

if __name__ == "__main__":
    generate_all_files(verbose=True)
