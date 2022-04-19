import numpy as np

# TODO: Tune this
DISTSQ_THRESHOLD = 16  # NOTE: this is Euclid distance squared (i.e. 16 = ~4 px of error allowed)


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


def findLargestCluster1D(data: np.ndarray,
                         max_diff=DISTSQ_THRESHOLD) -> np.ndarray:
    '''
    @brief Find largest "cluster" in 1D array by sorting and thresholding
    @param[in] data (N x 1) array of 1D data to find cluster
    @param[in] max_diff Maximum difference between values to count as cluster
    @return cluster_ind_mask Boolean mask showing which points in `data` are part of cluster 
    '''
    N = data.shape[0]
    data_sorted_ind = np.argsort(data)
    reverse_ind_map = np.arange(N)[data_sorted_ind]

    data_sorted = data[data_sorted_ind]
    data_sorted_diff = data_sorted[1:N] - data_sorted[0:N - 1]
    mask = data_sorted_diff > max_diff
    print(data_sorted, data_sorted_diff, mask)

    return mask[reverse_ind_map]


def rejectOutliersRansacDist(
        prev_coord: np.ndarray,
        new_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Find the largest 1D "cluster" of 1D points within certain threshold, and reject outliers accordingly
    '''

    # Use distance as form of thresholding
    deltas = (prev_coord - new_coord)
    deltas2 = deltas**2

    dist2 = np.sum(deltas2, axis=1)

    pruning_mask = findLargestCluster1D(dist2)

    pruned_prev_coord = prev_coord[pruning_mask]
    pruned_new_coord = new_coord[pruning_mask]

    return pruned_prev_coord, pruned_new_coord


def rejectOutliers(prev_old_coord: np.ndarray, prev_coord: np.ndarray,
                   new_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Reject outliers using combination of various methods
        
    @note  It is assumeed that the points correspond with each other
    
    @param[in] prev_old_coord   (K' x 2) Coordinates of features that prev_coord were tracked from [x,y]
    @param[in] prev_coord       (K x 2) Coordinates of features which are being tracked from [x,y]
    @param[in] new_coord        (K x 2) New coordinates of features which are tracked to [x,y]

    @note   It is possible that K' < K because features were appended. 
            Consider all appended features as valid because we have no prior

    @return pruned_prev_coord   (k x 2) Pruned previous coordinates
    @return pruned_new_coord    (k x 2) Pruned current/new coordinates
    '''

    np.savez("outlier_test.npy",
             prev_old_coord=prev_old_coord,
             prev_coord=prev_coord,
             new_coord=new_coord)

    good_old, good_new = rejectOutliersRadarGeometry(prev_old_coord,
                                                     prev_coord, new_coord)
    rejectOutliersRansacDist(good_old, good_new)

    return good_old, good_new


if __name__ == "__main__":
    with np.load("outlier_test.npy") as data:
        prev_coord = data["prev_coord"]
        new_coord = data["new_coord"]
        prev_old_coord = data["prev_old_coord"]

        # rejectOutliers(prev_old_coord, prev_coord, new_coord)
        rejectOutliersRansacDist(prev_coord, new_coord)