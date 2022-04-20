import numpy as np
from parseData import RANGE_RESOLUTION_CART_M  # m per px
import networkx as nx
from scipy.spatial.distance import cdist

from genFakeData import generateFakeFeatures, getRotationMatrix, generateFakeCorrespondences, plotFakeFeatures

# TODO: Tune this
DIST_THRESHOLD_M = 2.5  # why is the variance so fking high
DIST_THRESHOLD_PX = DIST_THRESHOLD_M / RANGE_RESOLUTION_CART_M  # Euclidean distance threshold
# NOTE: this is Euclid distance squared (i.e. 25 = ~5 px of error allowed)
DISTSQ_THRESHOLD_PX = DIST_THRESHOLD_PX * DIST_THRESHOLD_PX


def rejectOutliers(prev_old_coord: np.ndarray, prev_coord: np.ndarray,
                   new_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Reject outliers by using radar geometry to find dynamic/moving features
    
    @details For the first and second correspondence, form a graph G1 and G2, 
    where each point is a feature, and each edge is the distance between the 2 points.
    Because of radar geometry, the distance between any 2 points in G1 should be the 
    same (within threshold) as the same points in G2. 

    This is thus equivalent to forming an unweighted graph G, expressed as an adjacency matrix
    where if the difference in distance between i and j < thresh, then entry = 1, 0 otherwise.
    We then form the inlier set by finding the maximal clique in G.

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
    # TODO: Need to find Euclidean distance for every point with every other point
    # TODO: Use cdist
    dist1 = cdist(orig_prev_coord, prev_old_coord)
    dist2 = cdist(orig_new_coord, orig_prev_coord)

    # TODO: Form adj mat and graph

    # Return pruned coordinates
    pruned_prev_coord = prev_coord[pruning_mask]
    pruned_new_coord = new_coord[pruning_mask]

    return pruned_prev_coord, pruned_new_coord


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(
        f"Distance threshold: {DIST_THRESHOLD_M} [m] {DIST_THRESHOLD_PX:.2f} [px]"
    )

    n_points = 10
    theta_max_deg = 20
    max_translation_m = 3
    prev_old_coord, prev_coord, theta_deg1, h1 = generateFakeCorrespondences(
        n_points=n_points,
        theta_max_deg=theta_max_deg,
        max_translation_m=max_translation_m)

    print(f"First Transform\n\ttheta = {theta_deg1}\n\ttrans = {h1.flatten()}")

    prev_coord, new_coord, theta_deg2, h2 = generateFakeCorrespondences(
        prev_coord,
        n_points=n_points,
        theta_max_deg=theta_max_deg,
        max_translation_m=max_translation_m)

    print(f"Second Transform\n\ttheta = {theta_deg2}\n\ttrans = {h2.flatten()}")

    plotFakeFeatures(prev_old_coord, prev_coord, new_coord)
    plt.show()

    rejectOutliers(prev_old_coord, prev_coord, new_coord)
