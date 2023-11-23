import numpy as np

# function to check orbital occupations
def check_orbs(occnums_w, threshold):
    lorbs_ok = True
    norbs_ok = True
    # sum over the first two dimensions (spin and kpts)
    occs_sum = np.sum(occnums_w, axis=(0, 1))
    # check the l dimension
    occs_l = occs_sum[:, -1]
    if max(occs_l) > threshold:
        lorbs_ok = False
    occs_n = occs_sum[-1, :]
    if max(occs_n) > threshold:
        norbs_ok = False
    return lorbs_ok, norbs_ok
