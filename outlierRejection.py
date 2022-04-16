import numpy as np


def rejectOutliersRadarGeometry(
        prev_old_coord: np.ndarray, prev_coord: np.ndarray,
        new_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Reject outliers by using radar geometry to find dynamic/moving features: 
           By comparing distances between old and new correspondences, we can reject features
           whose distances between them are not constant, as this means that features are not
           static and are bad for tracking
    @note It is assumeed that the points correspond with each other
    
    @param[in] prev_old_coord   (K x 2) Coordinates of features that prev_coord were tracked from [x,y]
    @param[in] prev_coord       (K x 2) Coordinates of features which are being tracked from [x,y]
    @param[in] new_coord        (K x 2) New coordinates of features which are tracked to [x,y]

    @return pruned_prev_coord   (k x 2) Pruned previous coordinates
    @return pruned_new_coord    (k x 2) Pruned current/new coordinates
    '''
    assert prev_old_coord.shape == prev_coord.shape, "Coordinates should be the same shape"
    assert prev_coord.shape == new_coord.shape, "Coordinates should be the same shape"

    # TODO: Actually perform outlier rejection, now only returning the same coordinate back
    pruned_prev_coord = prev_coord 
    pruned_new_coord = new_coord

    return pruned_prev_coord, pruned_new_coord
