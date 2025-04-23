"""
This script investigates the mass relation between the stellar mass and the central BH mass.
Data is saved to a text file and the mass_relation.ipynb notebook is used to plot the data.

This script requires access to the full FABLE snapshots.

------------------------------------------------------------------------------------------------

Stephanie Buttigieg (sb2583@cam.ac.uk)

"""
import utils as util 
import numpy as np
import illustris_python.snapshot as snapshot
import illustris_python.groupcat as gc
import mpi4py.MPI as MPI

LITTLE_H = 0.679
basepath = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100' # modify basepath as needed

def generate_mass_data_serial(redshift, total_mass = False, verbose=False):
    """
    Generate the data for the mass relation between the stellar mass and the central BH mass.
    Parameters:
    redshift (float): The redshift at which to generate the data.
    total_mass (bool): If True, the total stellar mass of the galaxy within twice the half-mass radius is used. If False, the mass within the half-mass radius is used.
    verbose (bool): If True, print additional information.
    """

    scale_factors = util.get_scale_factors()
    redshifts = 1/scale_factors - 1

    snap = np.argmin(np.abs(redshifts - redshift)) # find the snapshot number closest to the desired redshift

    n_groups = gc.loadHeader(basepath, snap)['Ngroups_Total']
    if verbose:
        print('Total number of groups:', n_groups)
    if not total_mass:
        mass_field = 'SubhaloMassInHalfRadType'
    else:
        mass_field = 'SubhaloMassInRadType'
    subhalo_masses = gc.loadSubhalos(basepath, snap, fields=[mass_field])[:,4]

    first_subs = gc.loadHalos(basepath, snap, fields=['GroupFirstSub'])

    bh_masses = []
    host_masses = []
    for i in range(n_groups):
        if verbose:
            print('Group number:', i)
        # load all the BH particles in the subhalo and keep the mass of the most massive one
        group_first_sub = first_subs[i]
        local_bh_masses = snapshot.loadSubhalo(basepath, snap, group_first_sub, partType=5, fields=['BH_Mass'])
        if not isinstance(local_bh_masses, dict) and len(local_bh_masses) > 0:
            bh_mass = np.max(local_bh_masses)
            if bh_mass > 0:
                bh_masses.append(bh_mass)
                host_masses.append(subhalo_masses[group_first_sub])

    bh_masses = np.array(bh_masses)*1e10/LITTLE_H
    host_masses = np.array(host_masses)*1e10/LITTLE_H

    mask = np.where(host_masses > 0)

    bh_masses = bh_masses[mask]
    host_masses = host_masses[mask]

    # write bh_masses and host_masses to a text file
    np.savetxt(f'../data/bh_masses_{snap}_serial.txt', bh_masses)
    if not total_mass:
        np.savetxt(f'../data/host_masses_{snap}_serial.txt', host_masses)
    else:
        np.savetxt(f'../data/host_masses_total_{snap}_serial.txt', host_masses)

def generate_mass_data_parallel(redshift, total_mass=False, verbose=False):
    """
    Parallel version of generate_mass_data_serial.
    Each MPI process handles a subset of groups to extract BH and stellar masses.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only root does I/O and snapshot selection
    if rank == 0:
        scale_factors = util.get_scale_factors()
        redshifts = 1 / scale_factors - 1
        snap = np.argmin(np.abs(redshifts - redshift))

        header = gc.loadHeader(basepath, snap)
        n_groups = header['Ngroups_Total']
        if verbose:
            print('Total number of groups:', n_groups)

        mass_field = 'SubhaloMassInHalfRadType' if not total_mass else 'SubhaloMassInRadType'
        subhalo_data = gc.loadSubhalos(basepath, snap, fields=[mass_field, 'SubhaloLenType'])
        subhalo_masses = subhalo_data[mass_field][:, 4]
        subhalo_bh_count = subhalo_data['SubhaloLenType'][:, 5] # number of BHs in each subhalo
        first_subs = gc.loadHalos(basepath, snap, fields=['GroupFirstSub'])
    else:
        snap = None
        n_groups = None
        subhalo_masses = None
        first_subs = None
        subhalo_bh_count = None

    # Broadcast shared values
    snap = comm.bcast(snap, root=0)
    n_groups = comm.bcast(n_groups, root=0)
    first_subs = comm.bcast(first_subs, root=0)
    subhalo_masses = comm.bcast(subhalo_masses, root=0)
    subhalo_bh_count = comm.bcast(subhalo_bh_count, root=0)

    # Each rank processes a slice of groups
    local_bh_masses = []
    local_host_masses = []
    local_bh_ids = [] # to check for duplicates

    for i in range(rank, n_groups, size):
        if verbose:
            print(f"Rank {rank} processing group {i}")
        group_first_sub = first_subs[i]
        if group_first_sub != -1:
            number_of_bhs = subhalo_bh_count[group_first_sub]
            if number_of_bhs > 0:
                local_data = snapshot.loadSubhalo(basepath, snap, group_first_sub, partType=5, fields=['BH_Mass', 'ParticleIDs'])
                local_masses = local_data['BH_Mass']
                local_ids = local_data['ParticleIDs']
                if not isinstance(local_masses, dict) and len(local_masses) > 0:
                    bh_mass = np.max(local_masses) # only keep the most massive BH in the central subhalo
                    if bh_mass > 0:
                        bh_id = local_ids[np.argmax(local_masses)]
                        local_bh_masses.append(bh_mass)
                        local_host_masses.append(subhalo_masses[group_first_sub])
                        local_bh_ids.append(bh_id)

    # Gather all results to root
    all_bh_masses = comm.gather(local_bh_masses, root=0)
    all_host_masses = comm.gather(local_host_masses, root=0)
    all_bh_ids = comm.gather(local_bh_ids, root=0)

    if rank == 0:
        # Check for duplicates
        all_bh_ids = [item for sublist in all_bh_ids for item in sublist]
        unique_bh_ids = set(all_bh_ids)
        if len(all_bh_ids) != len(unique_bh_ids):
            print(f"Rank {rank} found duplicates in BH IDs.")
        # Check for duplicates in group numbers
       
        # Flatten the lists
        bh_masses = np.array([m for sublist in all_bh_masses for m in sublist]) * 1e10 / LITTLE_H
        host_masses = np.array([m for sublist in all_host_masses for m in sublist]) * 1e10 / LITTLE_H

        mask = host_masses > 0
        bh_masses = bh_masses[mask]
        host_masses = host_masses[mask]

        # Save to file
        np.savetxt(f'../data/bh_masses_{snap}.txt', bh_masses)
        if not total_mass:
            np.savetxt(f'../data/host_masses_{snap}.txt', host_masses)
        else:
            np.savetxt(f'../data/host_masses_total_{snap}.txt', host_masses)


if __name__ == '__main__':
    redshifts = [0, 2, 4]
    for redshift in redshifts:
        generate_mass_data_parallel(redshift=redshift, total_mass=True, verbose=False)