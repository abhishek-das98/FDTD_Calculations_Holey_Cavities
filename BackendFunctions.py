import numpy as np
from scipy.interpolate import griddata
from pathlib import Path

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

def make_far_field_plot_without_pol(phi, theta, vals, NA = 0.68, n=1.0, 
                                    basename = "Inverse_Design_Holey_Cavity_Far_Field_Overlap_cal_Unolarized", 
                                    save_fig=True, calculate_overlap=True):
    """
    Plots the far-field intensity distribution, optionally saving the figure and calculating the mode overlap and mentioning it on the plot.
    Parameters:
    phi, theta : 1D arrays (rad)
        Input angular coordinates
    vals : 2D array
        Far-field intensity values
    NA, n : float 
        Numerical aperture and refractive index (of air)
    basename : str
        Base name for saving the figure
    save_fig : bool
        Whether to save the figure
    calculate_overlap : bool
        Whether to calculate and display the mode overlap (Ideally when plotting the Gaussian beam, this should be False)
    """

    phi = np.asarray(phi)
    theta = np.asarray(theta)
    vals = np.asarray(vals)
    assert vals.shape == (len(theta), len(phi))

    if calculate_overlap:
        overlap = compute_gaussian_overlap_without_pol(phi, theta, vals, NA=NA, n=n)

    I = vals
    I_norm = I / (I.max() if I.max()>0 else 1.0)

    PHI, THETA = np.meshgrid(phi, theta, indexing = 'xy')

    theta_NA_deg = np.degrees(np.arcsin(min(NA/float(n), 1.0)))
    phi_circle = np.linspace(0, 2.0 * np.pi, 721)
    theta_circle = np.full_like(phi_circle, theta_NA_deg)

    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize = (6.5,5.5))
    pcm1 = ax1.pcolormesh(PHI, np.degrees(THETA), I_norm, shading='auto', cmap='jet')
    cbar1 = fig1.colorbar(pcm1, ax=ax1, label='Normalized Intensity')
    ax1.set_ylim(0, 90)
    ax1.set_title('Far_field: Normalized Intenity')
    ax1.plot(phi_circle, theta_circle, 'w--', lw = 2, label=f'NA={NA:.2f}, n = {n}')
    ax1.legend(loc='upper right')
    if calculate_overlap:
        ax1.text(0.02, 0.98, f'Mode overlap: {overlap:.2f}%', transform = ax1.transAxes, va = 'top', ha = 'left',
                bbox = dict(boxstyle = 'round, pad = 0.3', facecolor = 'black', alpha = 0.55), color = 'white')
        
    fig1.tight_layout()

    if save_fig:
        fn = f"{basename}_far_field.png"
        fig1.savefig(fn, dpi=300, bbox_inches='tight')
    plt.show()
    return
