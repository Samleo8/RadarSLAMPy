import numpy as np
from parseData import RANGE_RESOLUTION_CART_M  # m per px

N_FEATURES_BEFORE_RANSAC = 50

# TODO: Tune this
DIST_THRESHOLD_M = 3  # why is the variance so fking high
DIST_THRESHOLD_PX = DIST_THRESHOLD_M / RANGE_RESOLUTION_CART_M  # Euclidean distance threshold
# NOTE: this is Euclid distance squared (i.e. 25 = ~5 px of error allowed)
DISTSQ_THRESHOLD_PX = DIST_THRESHOLD_PX * DIST_THRESHOLD_PX

# For plotting/visualizing ransac
DO_PLOT = True
PLOT_FINAL_ONLY = True

PLOT_FINAL_ONLY &= DO_PLOT
if PLOT_FINAL_ONLY:
    DO_PLOT = False

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
    pruning_mask[:K_prev] = (dist_sq_diff > DISTSQ_THRESHOLD_PX)

    nRejected = K - np.count_nonzero(pruning_mask)
    print(f"Geometry-Based Outliers Rejected: {nRejected} ({100 * nRejected/K:.2f}%)")

    # Return pruned coordinates
    pruned_prev_coord = prev_coord[pruning_mask]
    pruned_new_coord = new_coord[pruning_mask]

    return pruned_prev_coord, pruned_new_coord


def findLargestCluster1D(data: np.ndarray,
                         max_diff=DISTSQ_THRESHOLD_PX) -> np.ndarray:
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
        new_coord: np.ndarray,
        n_iters: int = 20,
        n_try_points_percentage: float = 0.3,
        min_valid_percentage: float = 0.75,
        dist_thresh: float = DISTSQ_THRESHOLD_PX
) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Using RANSAC, find the best mean of 1D distances within certain threshold, and reject outliers accordingly
    @param[in] prev_coord   (K x 2) src/previous coordinate array, [x y] format
    @param[in] new_coord    (K x 2) target/current coordinate array, [x y] format

    @param[in] n_iters                  [RANSAC Param] Number of iterations to perform RANSAC
    @param[in] n_try_points_percentage  [RANSAC Param] Percentage of total points to try in RANSAC
    @param[in] n_try_points_percentage  [RANSAC Param] Percentage of total points to try in RANSAC

    @return prev_coord   (K' x 2) pruned previous coordinate array, [x y] format
    @return new_coord    (K' x 2) pruned coordinate array, [x y] format
    '''

    # Use distance as form of thresholding
    deltas = (prev_coord - new_coord)
    deltas2 = deltas**2

    dist2 = np.sum(deltas2, axis=1)

    # Basic RANSAC Algorithm
    # Want to find best "cluster" which has a mean of mu and variance/error of dist_thresh
    K = deltas.shape[0]
    n_try_points = int(n_try_points_percentage * K)
    min_valid_points = int(min_valid_percentage * K)

    if K < N_FEATURES_BEFORE_RANSAC:
        print("Too few features, will not perform distance RANSAC!")
        return prev_coord, new_coord

    if DO_PLOT:
        import matplotlib.pyplot as plt

    ind_range = np.arange(K)

    # Find best error and "model" (mean)
    bestError = np.inf
    bestMean = dist2.mean()

    for _ in range(n_iters):
        if DO_PLOT:
            plt.clf()

            # Background
            plt.scatter(ind_range,
                        dist2,
                        color='blue',
                        alpha=0.5,
                        label='full')

        # Find maybe inliners
        perm_ind = np.random.permutation(ind_range)
        maybeInliersInd = perm_ind[:n_try_points]
        maybeInliers = dist2[maybeInliersInd]

        # Get the all possible other inliners
        otherInliersInd = perm_ind[n_try_points:]
        otherInliers = dist2[otherInliersInd]

        # Find mu (which is the mean/"model" in this case)
        mu = maybeInliers.mean()
        mu_low = mu - dist_thresh
        mu_high = mu + dist_thresh
        thresh = (mu_low <= otherInliers) & (otherInliers <= mu_high)

        # Get actual other inliners
        otherInliersInd = otherInliersInd[thresh]
        otherInliers = dist2[otherInliersInd]

        nTotalInliers = otherInliersInd.shape[0] + n_try_points

        # Potentially valid model
        if nTotalInliers > min_valid_points:
            inlierPointsInd = np.hstack([maybeInliersInd, otherInliersInd])
            inlierPoints = dist2[inlierPointsInd]

            # Get the actually inlier points given the mean for figuring out the error
            newMean = inlierPoints.mean()

            mu_low = newMean - dist_thresh
            mu_high = newMean + dist_thresh
            thresh = (mu_low <= inlierPoints) & (inlierPoints <= mu_high)
            inlierPoints = dist2[inlierPointsInd[thresh]]

            newError = np.abs(inlierPoints - newMean).sum()

            if newError < bestError:
                bestError = newError
                bestMean = newMean

                print(f"Better mean {bestMean} found, with err {bestError}!")

        if DO_PLOT:
            plt.scatter(maybeInliersInd,
                        maybeInliers,
                        marker='+',
                        color='red',
                        label='maybeInliner')

            plt.scatter(otherInliersInd,
                        otherInliers,
                        marker='x',
                        color='green',
                        label='otherInliner')

            plt.axhline(y=mu,
                        color='red',
                        linestyle='dashed',
                        alpha=0.4,
                        label='mean')
            plt.axhline(y=mu_low,
                        color='red',
                        linestyle='dashed',
                        label='upper')
            plt.axhline(y=mu_high,
                        color='red',
                        linestyle='dashed',
                        label='lower')
            plt.legend()
            plt.tight_layout()

            plt.pause(0.01)

    # Obtain the actual inliers using the best mean
    mu_low = bestMean - dist_thresh
    mu_high = bestMean + dist_thresh
    pruning_mask = (mu_low <= dist2) & (dist2 <= mu_high)

    print("Final (actual) best mean:", dist2[pruning_mask].mean())

    if DO_PLOT or PLOT_FINAL_ONLY:
        import matplotlib.pyplot as plt

        print("Plotting final mean...")
        plt.close()
        plt.clf()

        # Background
        plt.scatter(ind_range, dist2, color='blue', alpha=0.5, label='full')

        # Final means
        plt.scatter(ind_range[pruning_mask],
                    dist2[pruning_mask],
                    color='green',
                    alpha=0.5,
                    label='final inliers')

        plt.axhline(y=bestMean,
                    color='green',
                    linestyle='dashed',
                    alpha=0.4,
                    label='mean')
        plt.axhline(y=mu_low, color='green', linestyle='dashed', label='upper')
        plt.axhline(y=mu_high, color='green', linestyle='dashed', label='lower')
        plt.tight_layout()
        plt.legend()

        plt.pause(0.01)

    pruned_prev_coord = prev_coord[pruning_mask]
    pruned_new_coord = new_coord[pruning_mask]

    # Calculate how many rejections
    nRejected = K - pruned_new_coord.shape[0]
    print(f"Dist-Based Outliers Rejected: {nRejected} ({100 * nRejected / K:.2f}%)")

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

    np.savez("outlier_test.npz",
             prev_old_coord=prev_old_coord,
             prev_coord=prev_coord,
             new_coord=new_coord)

    good_old, good_new = rejectOutliersRadarGeometry(prev_old_coord,
                                                     prev_coord, new_coord)
    good_old, good_new = rejectOutliersRansacDist(good_old, good_new, n_iters=20)

    return good_old, good_new


if __name__ == "__main__":
    print(f"Distance threshold: {DIST_THRESHOLD_M}[m] {DIST_THRESHOLD_PX}[px]")

    with np.load("outlier_test.npz") as data:
        prev_coord = data["prev_coord"]
        new_coord = data["new_coord"]
        prev_old_coord = data["prev_old_coord"]

        # rejectOutliers(prev_old_coord, prev_coord, new_coord)
        rejectOutliersRansacDist(prev_coord, new_coord)