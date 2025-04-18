import numpy as np

def get_scale_factors():
    """
    Read the scale factors from ExpansionList_128.
    """
    path_to_expansion_list = '/cosma/home/dp012/dc-butt3/BHMacroDelay/data/expansion_list.txt' # modify this as required
    scale_factors = []
    with open(path_to_expansion_list, 'r') as f:
        for line in f:
            line_data = line.split()
            scale_factors.append(float(line_data[0]))
    return np.array(scale_factors)