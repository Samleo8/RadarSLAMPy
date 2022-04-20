import numpy as np
from parseData import RANGE_RESOLUTION_CART_M  # m per px
import networkx as nx
from scipy.spatial.distance import cdist

def rejectOutliers(
        prev_old_coord: np.ndarray, prev_coord: np.ndarray,
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
    # TODO: Need to find euclidean distance for every point with every other point
    # TODO: Use cdist

    # TODO: Form adj mat and graph

    # Return pruned coordinates
    pruned_prev_coord = prev_coord[pruning_mask]
    pruned_new_coord = new_coord[pruning_mask]

    return pruned_prev_coord, pruned_new_coord

if __name__ == "__main__":
    print(f"Distance threshold: {DIST_THRESHOLD_M} [m] {DIST_THRESHOLD_PX:.2f} [px]")

    DO_PLOT = True
    # PLOT_FINAL_ONLY = True

    with np.load("outlier_test.npz") as data:
        prev_coord = data["prev_coord"]
        new_coord = data["new_coord"]
        prev_old_coord = data["prev_old_coord"]

        rejectOutliers(prev_old_coord, prev_coord, new_coord)