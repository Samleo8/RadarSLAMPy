import matplotlib.pyplot as plt
import numpy as np
from parseData import RANGE_RESOLUTION_CART_M  # m per px
import networkx as nx
from scipy.spatial.distance import cdist

from genFakeData import addNoise, createOutliers, generateFakeFeatures, getRotationMatrix, generateFakeCorrespondences, plotFakeFeatures

# TODO: Tune this
DIST_THRESHOLD_M = 2.5  # why is the variance so fking high
DIST_THRESHOLD_PX = DIST_THRESHOLD_M / RANGE_RESOLUTION_CART_M  # Euclidean distance threshold
# NOTE: this is Euclid distance squared (i.e. 25 = ~5 px of error allowed)
DISTSQ_THRESHOLD_PX = DIST_THRESHOLD_PX * DIST_THRESHOLD_PX

# Turn on when ready to test true outlier rejection
FORCE_OUTLIERS = True


def rejectOutliers(prev_coord: np.ndarray,
                   new_coord: np.ndarray, outlierInd=None) -> tuple[np.ndarray, np.ndarray]:
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
    pruning_mask = np.zeros(K, dtype=bool)

    # Obtain Euclidean distances between single point and every point in feature set
    # Should really only need to worry about diagonal
    dist_prev = cdist(prev_coord, prev_coord, metric='euclidean')
    dist_new = cdist(new_coord, new_coord, metric='euclidean')

    assert np.all(
        dist_prev.T == dist_prev), "Prev dist matrix should be symmetric"
    assert np.all(
        dist_new.T == dist_new), "New dist matrix should be symmetric"

    distDiff = np.abs(dist_prev - dist_new)

    distThreshMask = (distDiff <= DIST_THRESHOLD_PX).astype(np.int8)

    # Plot distDiff for visualization
    import matplotlib.pyplot as plt
    # plt.spy(distThreshMask)
    # plt.imshow(distDiff, cmap='hot', interpolation='nearest')
    # plt.show()

    # TODO: Use scipy sparse matrix?
    # Form graph using the distThreshMask matrix
    G = nx.Graph(distThreshMask)
    print("Finding largest clique", end='... ', flush=True)
    G_cliques = nx.find_cliques(G)

    # Find largest clique by looping through all found cliques
    # NOTE: Faster to loop through all cliques than to use maximum weight clique function
    bestClique = []
    bestCliqueSize = 0
    for clique in G_cliques:
        cliqueSize = len(clique)
        if cliqueSize > bestCliqueSize:
            bestCliqueSize = cliqueSize
            bestClique = clique

    pruning_mask[np.array(bestClique)] = True

    if outlierInd is not None:
        fullN = len(set(np.concatenate((bestClique, outlierInd))))
        assert (fullN == K), 'In perfect scenario, inliers and outliers should combine to form full set'

    print(f'Found clique of size {bestCliqueSize}!')

    # Return pruned coordinates
    pruned_prev_coord = prev_coord[pruning_mask]
    pruned_new_coord = new_coord[pruning_mask]

    return pruned_prev_coord, pruned_new_coord


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(
        f"Distance threshold: {DIST_THRESHOLD_M} [m] {DIST_THRESHOLD_PX:.2f} [px]"
    )

    np.random.seed(314159)

    n_points = 100
    n_outliers = int(n_points * 0.2)

    theta_max_deg = 20
    max_translation_m = 5

    prev_coord, new_coord, theta_deg, trans_vec = generateFakeCorrespondences(
        # prev_coord,
        n_points=n_points,
        theta_max_deg=theta_max_deg,
        max_translation_m=max_translation_m)

    # Force outliers
    if FORCE_OUTLIERS:
        new_coord_perfect = new_coord.copy()
        new_coord, outlier_ind = createOutliers(new_coord,
                                                n_outliers,
                                                noiseToAdd=DIST_THRESHOLD_PX *
                                                2)
        # Add some noise to the data as well to test thresholding
        # new_coord = addNoise(new_coord, variance=DIST_THRESHOLD_PX / 4) 
        print(outlier_ind, outlier_ind.shape)
    else:
        new_coord_perfect = None

    print(
        f"Transform\n\ttheta = {theta_deg:.2f} deg\n\ttrans = {trans_vec.flatten()} px"
    )

    pruned_prev_coord, pruned_new_coord = rejectOutliers(prev_coord, new_coord, outlier_ind)

    plotFakeFeatures(prev_coord, new_coord, alpha=0.1, show=False)

    plotFakeFeatures(None,
                     new_coord_perfect,
                     alpha=0.4,
                     title_append="(perfect)",
                     show=False)

    plotFakeFeatures(pruned_prev_coord,
                     pruned_new_coord,
                     title_append="(pruned)",
                     show=False)

    if FORCE_OUTLIERS:
        plotFakeFeatures(None,
                         None,
                         new_coord[outlier_ind],
                         title_append="(true outliers)",
                         alpha=0.8,
                         show=True)

    plt.show()