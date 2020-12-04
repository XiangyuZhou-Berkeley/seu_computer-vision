

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    for i in range(0, Hi):
        for j in range(0, Wi):
            new_i = i + pad_width0
            new_j = j + pad_width1
            for k in range(0, Hk):
                for l in range(0, Wk):
                    new_k = k - pad_width0
                    new_l = l - pad_width1
                    out[i, j] += padded[new_i - new_k, new_j - new_l] * kernel[k, l]

    ### END YOUR CODE

    return out
def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))
    ### YOUR CODE HERE
    k = size // 2
    t1 = 1 / (2 * np.pi * np.square(sigma))
    for i in range(0, size):
        for j in range(0, size):
            t2 = -(np.square(i - k) + np.square(j-k)) / (2 * np.square(sigma))
            kernel[i,j] = np.exp(t2) * t1
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array(
            [[0,0,0],
            [0.5,0,-0.5],
            [0,0,0]]
    )
    out = conv(img,kernel)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array(
            [[0, 0.5, 0],
            [0, 0, 0],
            [0, -0.5, 0]]
     )
    out = conv(img,kernel)
    ### END YOUR CODE
    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    H = img.shape[0]
    W = img.shape[1]
    theta = np.zeros(img.shape)
   
    ### YOUR CODE HERE
    p_x = partial_x(img)
    p_y = partial_y(img)
    G = np.sqrt(np.square(p_x) + np.square(p_y))
    theta = np.arctan2(p_y, p_x) * 180 / np.pi
    for i in range(0, H):
        for j in range(0, W):
            if theta[i,j] < 0:
                theta[i,j] = theta[i,j] + 360
           
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    ### BEGIN YOUR CODE
    #padding with (1,1) zero padding ,so it will be easy to handle the edge of the image 
    pad_width = ((1, 1), (1, 1))
    padded = np.pad(G, pad_width, mode='constant',constant_values = 0)
    for i in range(0, H):
        for j in range(0, W):
            if (theta[i, j] == 0 or theta[i, j] == 180 or theta[i,j] == 360):
                b1 = (padded[i + 1, j + 1] >= padded[i + 1, j]) and (padded[i + 1,j + 1] >= padded[i + 1,j + 2])
                if b1:
                    out[i, j] = padded[i + 1, j + 1]
            elif (theta[i, j] == 45 or theta[i,j] == 225):
                b1 = (padded[i + 1, j + 1] >= padded[i + 2, j + 2])and (padded[i + 1,j + 1] >= padded[i, j])
                if b1:
                    out[i, j] = padded[i + 1,j + 1]
            elif (theta[i,j] == 90 or theta[i,j] == 270):
                b1 = (padded[i + 1, j +1] >= padded[i + 2, j + 1]) and (padded[i + 1 ,j + 1] >= padded[i, j + 1])
                if b1:
                    out[i, j] = padded[i + 1, j + 1]
            elif (theta[i,j] == 135 or theta[i,j] == 315):
                b1 = (padded[i + 1, j + 1] >= padded[i, j + 2])and (padded[i + 1,j + 1] >= padded[i + 2, j])
                if b1:
                    out[i,j] = padded[i + 1,j + 1]                        
                   
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)
    H,W = img.shape
    ### YOUR CODE HERE
    strong_edges = img > high
    #weak_edges = img > low + img <= high 本来想用这个的，但是一直显示The truth value of an array with more than one element is ambiguous. Use a.any() or a.all(),不知道怎样解决，就用for先做
    for i in range(0,H):
        for j in range(0,W):
            if img[i,j] > low and img[i,j] <= high:
                weak_edges[i,j] = True
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    ### YOUR CODE HERE
    # whether the pixel has been visited
    visited = np.zeros((H, W), dtype=np.bool)
    queue = []
    H1,W1 = indices.shape
    ###first get all strong points in a queue
    for i in range(0, H1):
        y1 = indices[i,0]
        x1 = indices[i,1]
        queue.append((y1,x1))
    ### 首先弹出第一个，然受标记已经访问，然后找到紧邻的点，找到之后若是属于弱并且未被访问过则 标记为强并且加入队列，标记已经访问
    while len(queue) != 0:
        (y,x) = queue.pop(0)
        visited[y, x] = True
        neighbors = get_neighbors(y,x,H,W)
        for (y2,x2) in neighbors:
            if weak_edges[y2, x2] == True and visited[y2, x2] == False:
                edges[y2, x2] = True
                queue.append((y2, x2))
                visited[y2, x2] = True
         
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size,sigma)
    smooth = conv(img,kernel)
    G,theta = gradient(smooth)
    nmi = non_maximum_suppression(G,theta)
    strong_edges,weak_edges = double_thresholding(nmi,high,low)
    edge = link_edges(strong_edges,weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i in range(0, W):
        for j in range(0, H):
            if img[i,j] == True:
                r_t = i * sin_t + j * cos_t
                for k in range(0,180):
                    r_tp = int(np.ceil(r_t[k]) + diag_len)
                    accumulator[r_tp, k] +=1
    ### END YOUR CODE

    return accumulator, rhos, thetas
