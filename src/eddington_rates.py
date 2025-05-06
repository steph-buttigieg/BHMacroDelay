"""
This script calculates the Eddington ratios for black holes in merger events.
"""
import utils as util
import bh_mergers as bh
import illustris_python.snapshot as snapshot
import numpy as np
import os
import astropy.units as u
import astropy.constants as const
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

LITTLE_H = 0.679

def eddington_rate(M_BH):
    epsilon_r = 0.1
    M_BH = M_BH*u.Msun
    eddington_rate = 4*np.pi*const.G*M_BH*const.m_p/(0.1*const.sigma_T*const.c)
    return eddington_rate.to(u.Msun/u.Gyr).value

basepath = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100'
delay_path = '../data/generated_data/delay_data'

def get_fable_eddington_ratios():
    """
    Get the Eddington ratios for the black holes in the merger events.
    Returns
    -------
    None.
    """

    if rank == 0:
        files = os.listdir(delay_path)

        indices = []
        simulation_redshifts = []

        for file in files:
            with open(os.path.join(delay_path, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.split()
                    index1 = data[0]
                    index2 = data[1]
                    simulation_redshift = float(data[4])
                    merger_index = f'{index1}_{index2}'
                    if merger_index not in indices:
                        indices.append(merger_index)
                        simulation_redshifts.append(simulation_redshift)

    # split the indices and redshifts into chunks for each rank
        indices = np.array_split(indices, size)
        simulation_redshifts = np.array_split(simulation_redshifts, size)
    else:
        indices = None
        simulation_redshifts = None

    # broadcast the indices and redshifts to all ranks
    indices_local = comm.scatter(indices, root=0)
    simulation_redshifts_local = comm.scatter(simulation_redshifts, root=0)

    # First load all the accretion rates in the snapshot before the numerical merger from FABLE
    primary_eddington_ratios_local = []
    secondary_eddington_ratios_local = []

    scale_factors = util.get_scale_factors()
    redshifts = 1/scale_factors - 1

    for i,index in enumerate(indices_local):
        print(f'Rank {rank} processing index {index}')
        index_primary = int(index.split('_')[0])
        index_secondary = int(index.split('_')[1])
        indices_to_find = [index_primary, index_secondary]
        simulation_z = simulation_redshifts_local[i]

        # find the index of the first redshift larger than the simulation redshift
        snap_before_merger = np.where(redshifts > simulation_z)[0][-1]
        for j, index_to_find in enumerate(indices_to_find):
            group_index, subhalo_index = bh.get_bh_group_subhalo_nr(index_to_find, snap_before_merger)
            if subhalo_index is not None:
                if subhalo_index >= 0:
                    bh_data = snapshot.loadSubhalo(basePath = basepath, snapNum = snap_before_merger, id=subhalo_index, partType = 5, fields = ['BH_Mass', 'BH_Mdot', 'ParticleIDs'])
                elif group_index >= 0:
                    bh_data = snapshot.loadHalo(basePath = basepath, snapNum = snap_before_merger, id=group_index, partType = 5, fields = ['BH_Mass', 'BH_Mdot', 'ParticleIDs'])
                else:
                    bh_data = snapshot.loadSubset(basePath = basepath, snapNum = snap_before_merger, partType = 5, fields = ['BH_Mass', 'BH_Mdot', 'ParticleIDs'])
                
            # find the index of the BH in the data
            if index_to_find not in bh_data['ParticleIDs']:
               print(f'Warning: BH with ParticleID {index_to_find} not found in snapshot {snap_before_merger}.')
               print('group_index:', group_index)
               print('subhalo_index:', subhalo_index)
               print(bh_data)
               continue
            bh_index = np.where(bh_data['ParticleIDs'] == index_to_find)[0][0]
            bh_mass = bh_data['BH_Mass'][bh_index] * 1e10/LITTLE_H
            bh_mdot = bh_data['BH_Mdot'][bh_index] * 1e10/LITTLE_H/(0.978/LITTLE_H)**2

            bh_eddington_rate = eddington_rate(bh_mass)
            eddington_ratio = bh_mdot / bh_eddington_rate

            if j == 0:
                primary_eddington_ratios_local.append([index_primary, snap_before_merger, eddington_ratio])
            else:
                secondary_eddington_ratios_local.append([index_secondary, snap_before_merger, eddington_ratio])

    # gather the results from all ranks
    primary_eddington_ratios = comm.gather(primary_eddington_ratios_local, root=0)
    secondary_eddington_ratios = comm.gather(secondary_eddington_ratios_local, root=0)

    if rank == 0:
        primary_eddington_ratios = np.concatenate(primary_eddington_ratios)
        secondary_eddington_ratios = np.concatenate(secondary_eddington_ratios)

        # save the results to a file
        output_path_primary = '../data/generated_data/fable_eddington_ratios_primary.txt'
        output_path_secondary = '../data/generated_data/fable_eddington_ratios_secondary.txt'
        with open(output_path_primary, 'w') as f:
            for i in range(len(primary_eddington_ratios)):
                f.write(f'{primary_eddington_ratios[i][0]} {primary_eddington_ratios[i][1]} {primary_eddington_ratios[i][2]}\n')
        with open(output_path_secondary, 'w') as f:
            for i in range(len(secondary_eddington_ratios)):
                f.write(f'{secondary_eddington_ratios[i][0]} {secondary_eddington_ratios[i][1]} {secondary_eddington_ratios[i][2]}\n')

if __name__ == "__main__":
    get_fable_eddington_ratios()

