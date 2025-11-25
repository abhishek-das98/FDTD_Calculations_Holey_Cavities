import numpy as np
from scipy.interpolate import griddata

def gaussian_far_field_no_polarization(theta, phi, NA, n=1.0):

    """
    Compute an ideal far-field Gaussian beam profile without polarization effects
    given the theta and phi angles, numerical aperture (NA), and refractive index (n).
    Outputs the abosute value of the amplitude

    Parameters:
    phi, theta : 1D arrays (rad)
        Input angular coordinates
    NA, n : float 
        Numerical aperture and refractive index (of air)
    Returns:
    A : 2D array
        Ideal far-field amplitude profile
    """
    
    theta_vals = np.asarray(theta)
    phi_vals = np.asarray(phi)
    theta0 = np.arcsin(NA/float(n))
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    A = np.exp(-(theta_grid / theta0)**2)

    return A


def compute_gaussian_overlap_without_pol(
        phi, theta, vals, 
        NA = 0.68, n = 1.0
):
    """
    Compute the overlap between the simulated and ideal Gaussian beam profiles.
    --------------------------------------------------------------------------
    Overlap Equation (standard mode overlap definition for scalar fields):
    
                    | ∫ E_sim(theta,phi) E_gauss(theta,phi) dOmega |^2
        eta   =  -------------------------------------------------------------
                 ∫ |E_sim(theta,phi)|^2 dOmega  ∫ |E_gauss(theta,phi)|^2 dOmega

    where:
        - E_sim(theta,phi)   : simulated amplitude of the far-field, calculated by taking the square-root of the intensity
        - E_gauss      : ideal Gaussian far-field amplitude (In our case, a real value)
        - dOmega = sin(theta) dtheta dphi is the solid-angle measure
        - η is a normalized power overlap between 0 and 1

    Parameters:
    phi, theta : 1D arrays (rad)
        Input angular coordinates
    NA, n : float 
        Numerical aperture and refractive index (of air)
    Returns:
    percentage_overlap : float
        Percentage overlap between simulated and ideal Gaussian beam profiles
    """
    phi   = np.asarray(phi)
    theta = np.asarray(theta)
    vals  = np.asarray(vals)
    assert vals.shape == (len(theta), len(phi))

    
    # Generate the ideal Gaussian far-field pattern, with a certain NA
    E_gauss = gaussian_far_field_no_polarization(theta, phi, NA, n)
    I_gauss = E_gauss**2

    # Simulated intensity is given. Convert to field amplitude
    # We shall discard the phase information from the amplitude
    I_sim = vals
    E_sim = np.sqrt(I_sim)
    
    # Build the solid-angle grid
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # Compute differential angle elements
    dtheta = float(np.mean(np.diff(theta)))
    dphi   = float(np.mean(np.diff(phi)))

    # Solid angle weighting
    dOmega = np.sin(theta_grid) * dtheta * dphi

    # Compute the numerator
    overlap = np.sum(E_sim * E_gauss * dOmega)
    numerator   = np.abs(overlap)**2

    # Compute the denominator
    norm_sim    = np.sum(I_sim   * dOmega)
    norm_gauss  = np.sum(I_gauss * dOmega)
    denominator = norm_sim * norm_gauss

    # overlap efficiency and percentage
    eta = numerator / denominator
    percentage_overlap = float(100 * eta)
    return percentage_overlap
