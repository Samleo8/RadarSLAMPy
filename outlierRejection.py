import matplotlib.pyplot as plt
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


def rejectOutliers(prev_coord: np.ndarray,
                   new_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Reject outliers by using radar geometry to find dynamic/moving features
    
    @details For the first and second feature set, form a graph G1 and G2,
    where each point is a feature, and each edge is the distance between the 2 points.
    Because of radar geometry, the distance between any 2 points in G1 should be the 
    same (within threshold) as the same points in G2. 

    This is thus equivalent to forming an unweighted graph G, expressed as an adjacency matrix
    where if the difference in distance between i and j < thresh, then entry = 1, 0 otherwise.
    We then form the inlier set by finding the maximal clique in G.

    @note  It is assumeed that the points correspond with each other
    
    @param[in] prev_coord       (K x 2) Coordinates of features which are being tracked from [x,y]
    @param[in] new_coord        (K x 2) New coordinates of features which are tracked to [x,y]

    @return pruned_prev_coord   (k x 2) Pruned previous coordinates
    @return pruned_new_coord    (k x 2) Pruned current/new coordinates
    '''
    assert prev_coord.shape == new_coord.shape, "Coordinates should be the same shape"

    # Check for appended features by comparing lengths
    K = prev_coord.shape[0]
    pruning_mask = np.ones(K, dtype=bool)

    # Obtain Euclidean distances between single point and every point in feature set
    # Should really only need to worry about diagonal
    dist_prev = cdist(prev_coord, prev_coord, metric='euclidean')
    dist_new = cdist(new_coord, new_coord, metric='euclidean')

    assert np.all(dist_prev.T == dist_prev), "Prev dist matrix should be symmetric"
    assert np.all(dist_new.T == dist_new), "New dist matrix should be symmetric"

    distDiff = np.abs(dist_prev - dist_new)

    distThreshMask = distDiff <= DIST_THRESHOLD_PX

    # Plot distDiff for visualization
    import matplotlib.pyplot as plt
    plt.spy(distThreshMask)
    # plt.imshow(distDiff, cmap='hot', interpolation='nearest')
    plt.show()

    # TODO: Use scipy sparse matrix?
    G = nx.Graph(distDiff)

    # Return pruned coordinates
    pruned_prev_coord = prev_coord[pruning_mask]
    pruned_new_coord = new_coord[pruning_mask]

    return pruned_prev_coord, pruned_new_coord


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(
        f"Distance threshold: {DIST_THRESHOLD_M} [m] {DIST_THRESHOLD_PX:.2f} [px]"
    )

    np.random.seed(42)

    n_points = 10
    theta_max_deg = 0  # 20
    max_translation_m = 3
    prev_old_coord, prev_coord, theta_deg1, h1 = generateFakeCorrespondences(
        n_points=n_points,
        theta_max_deg=theta_max_deg,
        max_translation_m=max_translation_m)

    print(
        f"First Transform\n\ttheta = {theta_deg1:.2f} deg\n\ttrans = {h1.flatten()} px"
    )

    prev_coord_copy = prev_coord.copy()
    prev_coord, new_coord, theta_deg2, h2 = generateFakeCorrespondences(
        prev_coord,
        n_points=n_points,
        theta_max_deg=theta_max_deg,
        max_translation_m=max_translation_m)

    print(
        f"Second Transform\n\ttheta = {theta_deg2:.2f} deg\n\ttrans = {h2.flatten()} px"
    )

    pruned_prev_coord, pruned_new_coord = rejectOutliers(prev_coord, new_coord)

    plotFakeFeatures(prev_old_coord,
                     prev_coord,
                     new_coord,
                     alpha=0.1,
                     show=False)

    plotFakeFeatures(prev_old_coord,
                     prev_coord,
                     new_coord,
                     title_append="(pruned)",
                     show=True)
