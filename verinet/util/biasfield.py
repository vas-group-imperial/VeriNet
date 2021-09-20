import numpy as np

# Polynomial regression
def get_matrix(x, y, order):
    def get_list(order):
        if order  == 1:
            return [x, x * y, y, np.ones_like(x)]
        elif order == 2:
            return [x**2, x**2 * y,  x**2 * y**2, x * y**2, y**2] + get_list(1)
        elif order == 3:
            return [x**3, x**3 * y, x**3 * y**2, x**3 * y**3, x**2 * y**3, x * y**3, y**3] + get_list(2)
        elif order == 4:
            return [x**4, x**4 * y, x**4 * y**2, x**4 * y**3, x**4 * y**4, x**3 * y**4, x**2 * y**4, x * y**4, y**4] + get_list(3)
        else:
            raise RuntimeError("polynomial for order >= 5 not supported yet!")
    M = get_list(order)
    return np.array(M).T

def poly_fit(x, y, z, order):
    M = get_matrix(x.flatten(), y.flatten(), order)
    try:
        coeff, _, _, _ = np.linalg.lstsq(M, z.flatten())
    except:
        coeff = np.zeros(M.shape[1])
        coeff[-1] = 1
        print(coeff)
    return coeff

def get_idx(Npoints_x, Npoints_y=None):
    if Npoints_y is None:
        Npoints_y = Npoints_x

    x = np.linspace(0, 1, Npoints_x)
    y = np.linspace(0, 1, Npoints_y)
    X, Y = np.meshgrid(x, y)
    return X, Y

def rescale(x, vmin=0, vmax=1):
    tmp = x - np.min(x)
    tmp = tmp * (vmax - vmin) / np.max(tmp) + vmin
    return tmp

def get_normalized_coeffs(vmin=0, vmax=1, order=1):
    X, Y = get_idx(256)
    M = get_matrix(X.flatten(), Y.flatten(), order=order)
    coeff = np.ones((M.shape[1]))
    Z = np.reshape(np.dot(M, coeff), X.shape)
    Zscale = rescale(Z, vmin=vmin, vmax=vmax)
    coeff_scale = poly_fit(X, Y, Zscale, order)
    return coeff_scale