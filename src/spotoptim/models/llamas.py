import numpy as np


def llamas_pi(beta, x, k_lin=1e-3, penalty=1e3) -> np.ndarray:
    r"""
    Computes a superelliptic function for the upper part of a superellipse.
    This function models the superellipse using a horizontal line, an
    elliptical segment, and an almost vertical line.

    The superellipse formula is given by:

    \[ \left( \frac{w_V - w_{\text{zsl}}}{w_{\text{chl}} - w_{\text{zsl}}} \right)^{\text{cur}} +
       \left( \frac{\pi_V - \pi_{\text{chl}}}{\pi_{\text{zsl}} - \pi_{\text{chl}}} \right)^{\text{cur}} = 1 \]

    Args:
        beta (tuple): A tuple containing the coefficients (m_zsl, pi_zsl, m_chl, pi_chl, cur),
                      which define the shape and characteristics of the superellipse:

                        * m_zsl: x-coordinate for the horizontal line start
                        * pi_zsl: y-coordinate for the horizontal line
                        * m_chl: x-coordinate for the start of the vertical line
                        * pi_chl: y-coordinate of the center
                        * cur: curvature parameter defining the superellipse shape

        x (np.ndarray):
            An array of normalized values for `w_V` where the function is evaluated.
        k_lin (float, optional):
            The slope of the almost vertical line. The default is 1e-3.
        penalty (float, optional):
            A penalty value used when the function is not defined. The default is 1e3.

    Returns:
        np.ndarray:
            An array of calculated `pi_V` values corresponding to each `w_V` from the input `x`.

    Notes:
        *`k_lin` is a constant (0.001) used for the slope of the almost vertical line.

    Examples:
        >>> import numpy as np
            from spotoptim.models.llamas import llamas_pi
            beta = (0.2, 1.0, 0.8, 0.5, 2.0)
            x_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            llamas_pi(beta, x_values)
            array([  1.  ,   0.99300665,   0.9330127 ,   0.7763854 , -62.        ])
        >>> beta = np.ones(5)
            x_values = 0.1 * np.ones(5)
            print(llamas_pi(beta, x_values))
            array([1., 1., 1., 1., 1.])
        >>> print(llamas_pi(beta, x_values).shape)
            (5,)

    References:
        Llamas, X., and Eriksson, L. Control-oriented compressor model with adiabatic efficiency extrapolation. SAE International Journal of Engines 10, 4 (2017), 1903–1916.
        DOI:10.4271/2017-01-1032
    """
    m_zsl, pi_zsl, m_chl, pi_chl, cur = beta

    # Using NumPy vectorized operations
    # Precompute terms used multiple times
    denominator_m = m_chl - m_zsl
    pi_diff = pi_zsl - pi_chl

    # Initialize the result array
    full_pi_V = np.empty_like(x)

    # Horizontal line
    mask_horizontal = x < m_zsl
    full_pi_V[mask_horizontal] = pi_zsl

    # Elliptical part
    mask_ellipse = (x >= m_zsl) & (x <= m_chl)
    normalized_m = (x[mask_ellipse] - m_zsl) / denominator_m
    test_term = 1 - normalized_m**cur
    pi_V_ellipse = np.where(test_term < 0, penalty, pi_chl + pi_diff * test_term ** (1 / cur))
    full_pi_V[mask_ellipse] = pi_V_ellipse

    # Almost vertical line
    mask_vertical = x > m_chl
    pi_vertical = ((1 + k_lin) * pi_chl / k_lin) - (x[mask_vertical] * (pi_chl / (m_chl * k_lin)))
    full_pi_V[mask_vertical] = pi_vertical

    return full_pi_V


def llamas_m(beta, y, k_lin=1e-3, penalty=1e3) -> np.ndarray:
    r"""
    Superelliptical function for the upper part of the superellipse.
    Still needs to be cropped to the right arm

    \[ x = h + a * (1 - |(y - k)/b|^r)^(1/r) \]

    \[ w_V = m_zsl + ((m_chl - m_zsl) * (1 - ((pi_V - pi_chl) / (pi_zsl - pi_chl))**cur)**(1/cur)) \]

    * Left: a straight line with zero slope at y_max
    * Mid: right upper arm of the elliptic function
    * Right: a straight line with negative slope (close to infinity) starting in minimum of elliptic function

    Args:
        beta (list or numpy.ndarray):
            The parameter values [a, b, r, h, k].
            * a (float): Scaling factor for the x-coordinate.
            * b (float): Scaling factor for the y-coordinate.
            * r (float): Shape parameter determining the curvature of the function.
            * h (float): Horizontal shift parameter from x axis.
            * k (float): Vertical shift parameter from y axis.
        y (numpy.ndarray):
            The input y-coordinate(s) where the function will be evaluated.
        k_lin (float, optional):
            The slope of the almost vertical line. Llamas set this to 1e-2.
            Default is 1e-3.

    Returns:
        float or numpy.ndarray:
            The evaluated x-coordinate(s) corresponding to the input y-coordinate(s).
            Penalizes parameter convergence in problematic (numerical error) direction and
            auto cuts of anything above y > b + k

    Examples:
            >>> import numpy as np
                from spotoptim.models.llamas import llamas_m
                beta = (0.1, 1.0, 0.8, 0.5, 2.0)
                x_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
                llamas_m(beta, x_values)
                    array([0.8      , 0.8      , 0.8      , 0.7415606, 0.52     ])
            >>> beta = np.ones(5)
                x_values = 0.1 * np.ones(5)
                print(llamas_m(beta, x_values))
                    array([1., 1., 1., 1., 1.])

    """
    m_zsl, pi_zsl, m_chl, pi_chl, cur = beta
    pi_V = y

    def calculate_m(pi):
        if pi < pi_chl:
            # CHL, Horizontal Line
            return m_chl

        elif pi_chl <= pi <= pi_zsl:
            # Elliptical part
            test_term = 1 - ((pi - pi_chl) / (pi_zsl - pi_chl)) ** cur
            if test_term <= 0:
                print(f"llamas_m: Warning 3 Elliptical part: beta values are out of bounds. Returning penalty: {penalty}")
                return penalty
            return m_zsl + ((m_chl - m_zsl) * (test_term ** (1 / cur)))

        else:  # pi > pi_zsl
            # ZSL, steep line with positive slope
            return (((1 + k_lin) * m_zsl) / k_lin) - (m_zsl / (pi_zsl * k_lin)) * pi

    vectorized_calculation = np.vectorize(calculate_m)

    return vectorized_calculation(pi_V)
