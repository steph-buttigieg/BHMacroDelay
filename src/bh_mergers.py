import numpy as np
import h5py
import illustris_python.snapshot as snapshot
import illustris_python.groupcat as gc
import utils as util
import astropy.units as u

LITTLE_H = 0.679
basepath = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100'

def get_correct_mass(BH_id, scale_factor):
    mass_file_path = '/cosma7/data/dp012/dc-butt3/bh_masses.hdf5' # modify this as required
    with h5py.File(mass_file_path, 'r') as hf:
        data = np.array(hf[f'{int(BH_id)}'])
        scale_factors = data[:, 0]
        masses = data[:, 1]

    scale_factors = np.array(scale_factors)
    masses = np.array(masses)
    # sort according to the scale factors
    indices = np.argsort(scale_factors)
    scale_factors = scale_factors[indices]
    masses = masses[indices]

    #find the scale factors that bracket scale_factor
    index = np.searchsorted(scale_factors, scale_factor)
    if index == 0:
        m = masses[0]*1e10/LITTLE_H
    elif index == len(scale_factors):
        m = masses[-1]*1e10/LITTLE_H
    else:
        m = masses[index-1]*1e10/LITTLE_H

    return m


class MergerEvent():
    
    def __init__(self, a, id1, id2):
        """
        Parameters
        ----------
        a : float
            scale factor of the merger event
        id1 : int
            id of the most massive black hole
        id2 : int
            id of the second black hole
        mass1 : float
            mass of the most massive black hole
        mass2 : float
            mass of the second black hole

        """
        mass1 = get_correct_mass(id1, a) # in solar masses already.
        mass2 = get_correct_mass(id2, a)
        # setting mass1 to always be the most massive black hole
        if mass1 >= mass2:
            self.mass1 = mass1
            self.mass2 = mass2
            self.id1 = int(id1)
            self.id2 = int(id2)
        else:
            self.mass1 = mass2
            self.mass2 = mass1
            self.id1 = int(id2)
            self.id2 = int(id1)
        self.total_mass = self.mass1 + self.mass2
        self.formation_redshift = 1/float(a) - 1
        self.formation_scale_factor = float(a)
        self.snap_no = 0

        self.mass_ratio = self.mass2/self.mass1

        self.chirp_mass = (self.mass1*self.mass2)**(3/5)/(self.mass1+self.mass2)**(1/5)

        self.find_snapshot(util.get_scale_factors())
        self.set_binary_separation()

    def find_snapshot(self, scale_factors):
        """
        Finds the snapshot number of the snap before the merger event.

        Parameters
        ----------
        scale_factors : list
            list of scale factors of the snapshots
        """
        if self.formation_scale_factor > scale_factors[-1]:
            self.snap_no = len(scale_factors) - 1
        for i in range(len(scale_factors)-1):
            if self.formation_scale_factor > scale_factors[i] and self.formation_scale_factor < scale_factors[i+1]:
                self.snap_no = i

    def set_binary_separation(self):
        """
        Get the binary separation of the black holes.

        Parameters
        ----------
        bh_ids : dict
            dictionary with key equal to the index of the galaxy and value equal to a tuple containing the the IDs of the black holes and the Hsml
        """
        bh_data = snapshot.loadSubset(basepath, self.snap_no, 5, fields=['ParticleIDs', 'BH_Hsml'])
        bh_ids = bh_data['ParticleIDs']
        index = np.where(bh_ids == self.id1)[0][0]
        scale_factors = util.get_scale_factors()
        a = scale_factors[self.snap_no]
        hsml1 = bh_data['BH_Hsml'][index]*u.kpc*a/LITTLE_H
        self.binary_separation = hsml1

def get_bh_group_subhalo_nr(bh_id, snap):
    """
    Get the group and subhalo number of a black hole in a given snapshot.

    Parameters
    ----------
    bh_id : int
        ID of the black hole.
    snap : int
        Snapshot number.

    Returns
    -------
    tuple
        Group number and subhalo number of the black hole.
        These are equal to -1 if the black hole is in the snapshot but not in the group or subhalo.
        And are equal to None if the black hole is not in the snapshot.
    """
    BH_ids = snapshot.loadSubset(basePath=basepath, snapNum=snap, partType=5, fields='ParticleIDs')
    if bh_id in BH_ids:
        bh_index = np.where(BH_ids == bh_id)[0][0]
        group_data = gc.loadHalos(basePath=basepath, snapNum=snap, fields=['GroupFirstSub', 'GroupLenType', 'GroupNsubs'])
        group_len_type = group_data['GroupLenType']
        group_first_sub = group_data['GroupFirstSub']
        group_n_subs = group_data['GroupNsubs']
        group_len_bh = group_len_type[:,5]

        number_of_groups = len(group_len_bh)

        cumulative_len = np.cumsum(group_len_bh)
        group_index = np.searchsorted(cumulative_len, bh_index, side='right')
        if group_index >= number_of_groups:
            group_index = -1

        first_bh_index_in_group = cumulative_len[group_index - 1] if group_index > 0 else 0

        subhalo_len_type = gc.loadSubhalos(basePath=basepath, snapNum=snap, fields=['SubhaloLenType'])
        subhalo_len_bh = subhalo_len_type[:,5]

        group_subhalos_len_bh = subhalo_len_bh[group_first_sub[group_index]:group_first_sub[group_index] + group_n_subs[group_index]]

        bh_index_in_group = bh_index - first_bh_index_in_group
        subhalo_index = np.searchsorted(np.cumsum(group_subhalos_len_bh), bh_index_in_group, side='right')
        if subhalo_index >= group_n_subs[group_index]:
            subhalo_index = -1
        else:
            subhalo_index = group_first_sub[group_index] + subhalo_index

    else: # this is the case where the BH is not in the snapshot
        group_index = None
        subhalo_index = None

    return group_index, subhalo_index