"""

Modified from scikit-image slic method

Original code (C) scikit-image

Modifications (C) Benjamin Irving

See licence.txt for more details

"""
import asyncio
import functools
#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX
from cpython cimport bool

import numpy as np
cimport numpy as np

from skimage.util import regular_grid

from .helpers import GeneralizedGamma

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None,functools.partial(f, data={
    'image_zyx': kwargs.get('image_zyx'),
    'mask': kwargs.get('mask'),
    'segments': kwargs.get('segments'),
    'step': kwargs.get('step'),
    'max_iter': kwargs.get('max_iter'),
    'spacing': kwargs.get('spacing'),
    'slic_zero': kwargs.get('slic_zero'),
    'only_dist': kwargs.get('only_dist')
}))

    return wrapped

# @background
def _slic_cython(double[:, :, :, ::1] image_zyx,
                 int[:, :, ::1] mask,
                 double[:, ::1] segments,
                 float step,
                 Py_ssize_t max_iter,
                 double[::1] spacing,
                 bint slic_zero,
                 bint only_dist):

    """Helper function for SLIC maskslic.

    Parameters
    ----------
    image_zyx : 4D array of double, shape (Z, Y, X, C)
        The input image.
    segments : 2D array of double, shape (N, 3 + C)
        The initial centroids obtained by SLIC as [Z, Y, X, C...].
    step : double
        The size of the step between two seeds in voxels.
    max_iter : int
        The maximum number of k-means iterations.
    spacing : 1D array of double, shape (3,)
        The voxel spacing along each image dimension. This parameter
        controls the weights of the distances along z, y, and x during
        k-means clustering.
    slic_zero : bool
        True to run SLIC-zero, False to run original SLIC.

    Returns
    -------
    nearest_segments : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.

    Notes
    -----
    The image is considered to be in (z, y, x) order, which can be
    surprising. More commonly, the order (x, y, z) is used. However,
    in 3D image analysis, 'z' is usually the "special" dimension, with,
    for example, a different effective resolution than the other two
    axes. Therefore, x and y are often processed together, or viewed as
    a cut-plane through the volume. So, if the order was (x, y, z) and
    we wanted to look at the 5th cut plane, we would write::

        my_z_plane = img3d[:, :, 5]

    but, assuming a C-contiguous array, this would grab a discontiguous
    slice of memory, which is bad for performance. In contrast, if we
    see the image as (z, y, x) ordered, we would do::

        my_z_plane = img3d[5]

    and get back a contiguous block of memory. This is better both for
    performance and for readability.
    """

    # initialize on grid
    cdef Py_ssize_t depth, height, width
    depth = image_zyx.shape[0]
    height = image_zyx.shape[1]
    width = image_zyx.shape[2]

    cdef Py_ssize_t n_segments = segments.shape[0]
    # number of features [X, Y, Z, ...]
    cdef Py_ssize_t n_features = segments.shape[1]

    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_z, step_y, step_x
    slices = regular_grid((depth, height, width), n_segments)
    step_z, step_y, step_x = [int(s.step if s.step is not None else 1)
                              for s in slices]

    cdef int[:, :, ::1] nearest_segments \
        = -1 * np.ones((depth, height, width), dtype=np.int32)
    cdef double[:, :, ::1] distance \
        = np.empty((depth, height, width), dtype=np.double)
    cdef Py_ssize_t[::1] n_segment_elems = np.zeros(n_segments, dtype=np.intp)

    cdef Py_ssize_t i, c, k, x, y, z, x_min, x_max, y_min, y_max, z_min, z_max
    cdef char change
    cdef double dist_center, cx, cy, cz, dy, dz

    cdef double sz, sy, sx
    sz = spacing[0]
    sy = spacing[1]
    sx = spacing[2]

    # The colors are scaled before being passed to _slic_cython so
    # max_color_sq can be initialised as all ones
    cdef double[::1] max_dist_color = np.ones(n_segments, dtype=np.double)
    cdef double dist_color

    # The reference implementation (Achanta et al.) calls this invxywt
    cdef double spatial_weight = float(1) / (step ** 2)
    spatial_weight = 0.0001

    print(f'Spatial weight: {spatial_weight}')

    for i in range(max_iter):
        print(f'Iteração {i}')
        change = 0
        distance[:, :, :] = 0.0 #DBL_MAX

        # assign pixels to segments
        for k in range(n_segments):

            # segment coordinate centers
            cz = segments[k, 0]
            cy = segments[k, 1]
            cx = segments[k, 2]

            # compute windows
            z_min = <Py_ssize_t>max(cz - 2 * step_z, 0)
            z_max = <Py_ssize_t>min(cz + 2 * step_z + 1, depth)
            y_min = <Py_ssize_t>max(cy - 2 * step_y, 0)
            y_max = <Py_ssize_t>min(cy + 2 * step_y + 1, height)
            x_min = <Py_ssize_t>max(cx - 2 * step_x, 0)
            x_max = <Py_ssize_t>min(cx + 2 * step_x + 1, width)
            segment_mask = nearest_segments == int(k)
            current_segment = image_zyx[segment_mask]
            # print('segmento:')
            # print(np.asarray(current_segment))
            gg = GeneralizedGamma(current_segment)

            for z in range(z_min, z_max):
                dz = (sz * (cz - z)) ** 2
                for y in range(y_min, y_max):
                    dy = (sy * (cy - y)) ** 2
                    for x in range(x_min, x_max):
                        # if np.asarray(image_zyx[z,y,x])[0] > 100.0:
                        #     print(np.asarray(image_zyx[z,y,x])[0])


                        if mask[z, y, x] == 0:
                            nearest_segments[z, y, x] = -1
                            continue

                        # print((dz + dy + (sx * (cx - x))))
                        # distance_sum = (dz + dy + (sx * (cx - x)))
                        # try:
                        #     dist_center = (1 - np.exp(-1/distance_sum)) * spatial_weight
                        # except:
                        #     dist_center = (1 - np.exp(-1/0.01)) * spatial_weight
                        spatial_distance = np.sqrt((cx-x)**2 + (cy-y)**2)
                        dist_center = (spatial_distance/sx)**2 * spatial_weight

                        try:
                            dist_center = (1 - np.exp(-1/dist_center)) * spatial_weight
                        except:
                            dist_center = (1 - np.exp(-1/0.01)) * spatial_weight

                        frequence = gg.function_value(np.asarray(image_zyx[z, y, x])[0])
                        dist_color = (1-np.exp(-1*frequence)) * (1 - spatial_weight)
                        # dist_color = frequence  * (1 - spatial_weight)
                        # print(f'Dist_center: {dist_center}, Dist_color: {dist_color}, Frequence: {frequence}')

                        # print(dist_color)
                        # dist_color = gg.function_value(np.asarray(image_zyx[z, y, x])[0])#(1 - spatial_weight) * (1-np.exp(-1*gg.function_value(np.asarray(image_zyx[z, y, x])[0])))
                        if slic_zero:
                            # TODO not implemented yet for slico
                            dist_center += dist_color / max_dist_color[k]
                        else:
                            if not only_dist:
                                if True:
                                    dist_center += dist_color
                                else:
                                    print(dist_color)
                                # else:
                                #     print('PRIMEIRA ITERAÇÃO')
                                #     dist_center = distance_sum**2 * spatial_weight
                                # print(dist_center > 0 and dist_center < 1)

                        # dist_center = dist_color

                        current_distance = np.asarray(distance[z, y, x])

                        #assign new distance and new label to voxel if closer than other voxels
                        if current_distance < dist_center:
                            print(f'Distância atual: {current_distance}. Distância calculada: {dist_center}')
                            print(f'Estou na iteração {i} e o valor do pixel mudou da classe {nearest_segments[z,y,x]} para a classe {k}')
                            nearest_segments[z, y, x] = int(k)
                            distance[z, y, x] = dist_center
                            #record change
                            change = 1

        # stop if no pixel changed its segment
        if change == 0:
            break

        # recompute segment centers

        # sum features for all segments
        n_segment_elems[:] = 0
        segments[:, :] = 0
        for z in range(depth):
            for y in range(height):
                for x in range(width):

                    if mask[z, y, x] == 0:
                        continue

                    if nearest_segments[z, y, x] == -1:
                        continue

                    k = nearest_segments[z, y, x]

                    n_segment_elems[k] += 1
                    segments[k, 0] += z
                    segments[k, 1] += y
                    segments[k, 2] += x
                    for c in range(3, n_features):
                        segments[k, c] += image_zyx[z, y, x, c - 3]

        # divide by number of elements per segment to obtain mean
        for k in range(n_segments):
            for c in range(n_features):
                segments[k, c] /= n_segment_elems[k]

        # If in SLICO mode, update the color distance maxima
        if slic_zero:
            for z in range(depth):
                for y in range(height):
                    for x in range(width):

                        if mask[z, y, x] == 0:
                            continue

                        if nearest_segments[z, y, x] == -1:
                            continue

                        k = nearest_segments[z, y, x]
                        dist_color = 0

                        for c in range(3, n_features):
                            dist_color += (image_zyx[z, y, x, c - 3] -
                                            segments[k, c]) ** 2

                        # The reference implementation seems to only change
                        # the color if it increases from previous iteration
                        if max_dist_color[k] < dist_color:
                            max_dist_color[k] = dist_color

    print(np.asarray(distance))
    print('argmax:')
    print(np.argmax(np.asarray(distance)[0][0]))
    print(len(np.asarray(distance)[0][0]))
    print(np.asarray(nearest_segments))

    return np.asarray(nearest_segments)


def _enforce_label_connectivity_cython(int[:, :, ::1] segments,
                                       int[:, :, ::1] mask,
                                       Py_ssize_t n_segments,
                                       Py_ssize_t min_size,
                                       Py_ssize_t max_size):
    """ Helper function to remove small disconnected regions from the labels

    Parameters
    ----------
    segments : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.
    n_segments: int
        Number of specified segments
    min_size: int
        Minimum size of the segment
    max_size: int
        Maximum size of the segment. This is done for performance reasons,
        to pre-allocate a sufficiently large array for the breadth first search
    Returns
    -------
    connected_segments : 3D array of int, shape (Z, Y, X)
        A label field with connected labels starting at label=1
    """

    # get image dimensions
    cdef Py_ssize_t depth, height, width
    depth = segments.shape[0]
    height = segments.shape[1]
    width = segments.shape[2]

    # neighborhood arrays
    cdef Py_ssize_t[::1] ddx = np.array((1, -1, 0, 0, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddy = np.array((0, 0, 1, -1, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddz = np.array((0, 0, 0, 0, 1, -1), dtype=np.intp)

    # new object with connected segments initialized to -1
    cdef Py_ssize_t[:, :, ::1] connected_segments \
        = -1 * np.ones_like(segments, dtype=np.intp)

    cdef Py_ssize_t current_new_label = 0
    cdef Py_ssize_t label = 0

    # variables for the breadth first search
    cdef Py_ssize_t current_segment_size = 1
    cdef Py_ssize_t bfs_visited = 0
    cdef Py_ssize_t adjacent

    cdef Py_ssize_t zz, yy, xx

    cdef Py_ssize_t[:, ::1] coord_list = np.zeros((max_size, 3), dtype=np.intp)

    # loop through all image
    with nogil:
        for z in range(depth):
            for y in range(height):
                for x in range(width):

                    if mask[z, y, x] == 0:
                        continue

                    if connected_segments[z, y, x] >= 0:
                        continue

                    # find the component size
                    adjacent = 0
                    label = segments[z, y, x]
                    #return connected segments
                    connected_segments[z, y, x] = current_new_label
                    current_segment_size = 1
                    bfs_visited = 0
                    coord_list[bfs_visited, 0] = z
                    coord_list[bfs_visited, 1] = y
                    coord_list[bfs_visited, 2] = x

                    #perform a breadth first search to find
                    # the size of the connected component
                    while bfs_visited < current_segment_size < max_size:
                        for i in range(6):
                            #six connected neighbours of the voxel
                            zz = coord_list[bfs_visited, 0] + ddz[i]
                            yy = coord_list[bfs_visited, 1] + ddy[i]
                            xx = coord_list[bfs_visited, 2] + ddx[i]

                            #check that within image
                            if (0 <= xx < width and
                                    0 <= yy < height and
                                    0 <= zz < depth):
                                #look
                                if (segments[zz, yy, xx] == label and
                                        connected_segments[zz, yy, xx] == -1):
                                    connected_segments[zz, yy, xx] = \
                                        current_new_label
                                    coord_list[current_segment_size, 0] = zz
                                    coord_list[current_segment_size, 1] = yy
                                    coord_list[current_segment_size, 2] = xx
                                    current_segment_size += 1
                                    if current_segment_size >= max_size:
                                        break
                                elif (connected_segments[zz, yy, xx] >= 0 and
                                      connected_segments[zz, yy, xx] != current_new_label):
                                    adjacent = connected_segments[zz, yy, xx]
                        bfs_visited += 1

                    # change to an adjacent one, like in the original paper
                    if current_segment_size < min_size:
                        for i in range(current_segment_size):
                            connected_segments[coord_list[i, 0],
                                               coord_list[i, 1],
                                               coord_list[i, 2]] = adjacent
                    else:
                        current_new_label += 1

    return np.asarray(connected_segments)

def _find_adjacency_map(int[:, :, ::1] segments):
    """

    @param segments: slic labelled image
    @return:
    border_mat = labels that border the first or last axial slice
    all_border_mat = labels that border any edge (To do)

    """

    # get image dimensions
    cdef Py_ssize_t depth, height, width
    depth = segments.shape[0]
    height = segments.shape[1]
    width = segments.shape[2]

    # neighborhood arrays (Py_ssize_t is the proper python definition for array indices)
    cdef Py_ssize_t[::1] ddx = np.array((1, -1, 0, 0, 0, 0))
    cdef Py_ssize_t[::1] ddy = np.array((0, 0, 1, -1, 0, 0))
    cdef Py_ssize_t[::1] ddz = np.array((0, 0, 0, 0, 1, -1))

    cdef Py_ssize_t zz, yy, xx
    cdef Py_ssize_t z, y, x
    cdef Py_ssize_t label = 0
    cdef Py_ssize_t neigh_lab = 0

    #create adjacency matrix
    cdef Py_ssize_t max_lab
    max_lab = np.max(segments)
    cdef Py_ssize_t[:, ::1] adj_mat = np.zeros((max_lab + 1, max_lab + 1), dtype=np.intp)
    cdef Py_ssize_t[::1] border_mat = np.zeros((max_lab +1), dtype=np.intp)

    # cdef Py_ssize_t marker1 = 1

    # loop through all image
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # find the component size
                label = segments[z, y, x]

                # Checking connectivity
                for i in range(6):
                    #six connected neighbours of the voxel
                    zz = z + ddz[i]
                    yy = y + ddy[i]
                    xx = x + ddx[i]

                    #check that within image
                    if 0 <= xx < width and 0 <= yy < height and 0 <= zz < depth:
                        #look
                        if segments[zz, yy, xx] != label:
                            neigh_lab = segments[zz, yy, xx]
                            adj_mat[label, neigh_lab] = 1
                            adj_mat[neigh_lab, label] = 1

                # Checking for border supervoxels
                if xx == 0 or xx == width-1:
                    border_mat[label] = 1

    return np.asarray(adj_mat), np.asarray(border_mat)

def _find_adjacency_map_mask(np.ndarray[np.int32_t, ndim=3] segmentsnp):
    """

    @param segments: slic labelled image
    @return:
    border_mat = labels that border the mask
    """

    # get image dimensions
    cdef Py_ssize_t depth, height, width
    depth = segmentsnp.shape[0]
    height = segmentsnp.shape[1]
    width = segmentsnp.shape[2]

    # neighborhood arrays (Py_ssize_t is the proper python definition for array indices)
    cdef Py_ssize_t[::1] ddx = np.array((1, -1, 0, 0, 0, 0))
    cdef Py_ssize_t[::1] ddy = np.array((0, 0, 1, -1, 0, 0))
    cdef Py_ssize_t[::1] ddz = np.array((0, 0, 0, 0, 1, -1))

    cdef Py_ssize_t zz, yy, xx
    cdef Py_ssize_t z, y, x
    cdef Py_ssize_t label = 0
    cdef Py_ssize_t neigh_lab = 0

    #create adjacency matrix
    cdef Py_ssize_t max_lab
    max_lab = segmentsnp.max()
    cdef Py_ssize_t[:, ::1] adj_mat = np.zeros((max_lab + 1, max_lab + 1), dtype=np.intp)
    cdef Py_ssize_t[::1] border_mat = np.zeros((max_lab +1), dtype=np.intp)

    cdef int[:, :, :] segments = segmentsnp

    # cdef Py_ssize_t marker1 = 1

    # loop through all image
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # find the component size
                label = segments[z, y, x]

                if label == -1:
                    # label is a background region and skip
                    continue

                # Checking connectivity
                for i in range(6):
                    #six connected neighbours of the voxel
                    zz = z + ddz[i]
                    yy = y + ddy[i]
                    xx = x + ddx[i]

                    #check that within image
                    if 0 <= xx < width and 0 <= yy < height and 0 <= zz < depth:

                        # Checking if the supervoxel borders with a background label
                        if segments[zz, yy, xx] == -1:
                            border_mat[label] = 1

                        elif segments[zz, yy, xx] != label:
                            neigh_lab = segments[zz, yy, xx]
                            adj_mat[label, neigh_lab] = 1
                            adj_mat[neigh_lab, label] = 1

    return np.asarray(adj_mat), np.asarray(border_mat)
