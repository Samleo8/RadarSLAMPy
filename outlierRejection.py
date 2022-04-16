import numpy as np

DISTSQ_THRESHOLD = 10 # NOTE: this is Euclid distance squared

def rejectOutliersRadarGeometry(
        prev_old_coord: np.ndarray, prev_coord: np.ndarray,
        new_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Reject outliers by using radar geometry to find dynamic/moving features: 
           By comparing distances between old and new correspondences, we can reject features
           whose distances between them are not constant, as this means that features are not
           static and are bad for tracking
    @note  It is assumeed that the points correspond with each other
    
    @param[in] prev_old_coord   (K' x 2) Coordinates of features that prev_coord were tracked from [x,y]
    @param[in] prev_coord       (K x 2) Coordinates of features which are being tracked from [x,y]
    @param[in] new_coord        (K x 2) New coordinates of features which are tracked to [x,y]

    @note   It is possible that K' < K because features were appended. 
            Consider all appended features as valid because we have no prior

    @return pruned_prev_coord   (k x 2) Pruned previous coordinates
    @return pruned_new_coord    (k x 2) Pruned current/new coordinates
    '''
    assert prev_coord.shape == new_coord.shape, "Coordinates should be the same shape"

    # Check for appended features by comparing lengths
    K_prev = prev_old_coord.shape[0]
    K = prev_coord.shape[0]
    assert K_prev <= K, f"There should only be appended or same number of features, not less ({K_prev} > {K})"

    pruning_mask = np.ones(K, dtype=bool)

    # TODO: Actually perform outlier rejection, now only returning the same coordinate back

    # Ensures that pruning only is done on non-appended features
    orig_prev_coord = prev_coord.copy()[:K_prev, :]
    orig_new_coord = new_coord.copy()[:K_prev, :]

    # Obtain Euclidean distances between coordinate correspondences
    dist1sq = (orig_prev_coord - prev_old_coord)**2
    dist1sq = np.sum(dist1sq, axis=1)  # (K, )

    dist2sq = (orig_new_coord - orig_prev_coord)**2
    dist2sq = np.sum(dist2sq, axis=1)  # (K, )

    # Difference in distances
    dist_sq_diff = np.abs(dist1sq - dist2sq)
    pruning_mask[:K_prev] = (dist_sq_diff > DISTSQ_THRESHOLD)

    nRejected = K - np.count_nonzero(pruning_mask)
    print(f"Outliers Rejected: {nRejected} ({100 * nRejected/K:.2f}%)")

    # Return pruned coordinates
    pruned_prev_coord = prev_coord[pruning_mask]
    pruned_new_coord = new_coord[pruning_mask]

    return pruned_prev_coord, pruned_new_coord
