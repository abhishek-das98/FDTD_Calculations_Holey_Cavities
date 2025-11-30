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
        NA = 0.68, n = 1.0,
        theta_max_deg = None  
):
    """
    Compute the mode-overlap between the simulated far-field intensity (vals)
    and an ideal Gaussian far-field WITHOUT polarization effects.

    Overlap is computed as:
        |∫ E_sim E_gauss dOmega|^2 / ( ∫|E_sim|^2 dOmega  ∫|E_gauss|^2 dOmega )

    Added FEATURE:
        If theta_max_deg is provided (in degrees), the integrals are evaluated
        ONLY for theta <= theta_max_deg.

    Parameters
    ----------
    phi, theta : 1D arrays (rad)
        Angular sampling coordinates of the simulated far-field.
    vals : 2D array
        Simulated intensity map, shape = (len(theta), len(phi)).
    NA, n : floats
        Gaussian beam NA and refractive index.
    theta_max_deg : float or None
        Upper limit of theta (in degrees) for restricting the overlap region.
        If None → use entire theta-domain.

    Returns
    -------
    percentage_overlap : float
        Mode overlap (0–100%).
    """

    # Convert inputs to arrays
    phi   = np.asarray(phi)
    theta = np.asarray(theta)
    vals  = np.asarray(vals)

    assert vals.shape == (len(theta), len(phi)), \
        f"vals has shape {vals.shape}, but expected {(len(theta), len(phi))}"

    # Build ideal Gaussian amplitude and its intensity
    E_gauss = gaussian_far_field_no_polarization(theta, phi, NA, n)
    I_gauss = E_gauss**2

    # Sim amplitude = sqrt(intensity)
    I_sim = vals
    E_sim = np.sqrt(I_sim)

    # Meshgrid for solid-angle element
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # Solid angle element (assume uniform grid)
    dtheta = float(np.mean(np.diff(theta)))
    dphi   = float(np.mean(np.diff(phi)))
    dOmega = np.sin(theta_grid) * dtheta * dphi

    # --------------------------------------
    #  Mask to restrict region if requested
    # --------------------------------------
    if theta_max_deg is not None:
        theta_max_rad = np.radians(theta_max_deg)
        mask = theta_grid <= theta_max_rad
    else:
        mask = np.ones_like(theta_grid, dtype=bool)

    # Apply mask
    E_sim_m   = E_sim[mask]
    E_gauss_m = E_gauss[mask]
    I_sim_m   = I_sim[mask]
    I_gauss_m = I_gauss[mask]
    dOmega_m  = dOmega[mask]

    # --------------------------
    #  Mode overlap numerator
    # --------------------------
    overlap = np.sum(E_sim_m * E_gauss_m * dOmega_m)
    numerator = np.abs(overlap)**2

    # --------------------------
    #  Normalization denominator
    # --------------------------
    norm_sim   = np.sum(I_sim_m   * dOmega_m)
    norm_gauss = np.sum(I_gauss_m * dOmega_m)
    denominator = norm_sim * norm_gauss

    eta = numerator / denominator if denominator > 0 else 0.0
    return float(100 * eta)


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

def intensity_overlap(I1, I2, theta, phi, theta_max_deg=None):
    """
    Compute the overlap between TWO intensity patterns
    over the theta-phi region, using solid-angle weighting.

    If theta_max_deg is provided, the overlap is computed only for
    theta <= theta_max_deg (in degrees). If theta_max_deg is None,
    the full theta range is used.

    Parameters
    ----------
    I1, I2 : 2D arrays
        Intensity maps with shape (len(theta), len(phi)).
    theta, phi : 1D arrays (rad)
        Angular coordinates.
        theta = polar angle (0 = optical axis),
        phi   = azimuthal angle.
    theta_max_deg : float or None, optional
        Upper limit of theta (in degrees) for restricting the overlap.
        If None, the entire theta-domain is used.

    Returns
    -------
    overlap : float
        Overlap between the intensity patterns.
        1   → intensities identical up to a scale factor
        0   → no correlation
    """

    # Convert inputs to arrays
    I1    = np.asarray(I1)
    I2    = np.asarray(I2)
    theta = np.asarray(theta)
    phi   = np.asarray(phi)

    # Ensure shapes match the expected grid structure
    assert I1.shape == I2.shape == (len(theta), len(phi)), \
        f"Intensity shapes must match (len(theta), len(phi)). Got {I1.shape} and {I2.shape}."

    # Build 2D angular grids for solid-angle weighting
    # indexing='ij' → first index = theta, second index = phi
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # Compute dθ and dφ (assumed uniform grids)
    dtheta = float(np.mean(np.diff(theta)))
    dphi   = float(np.mean(np.diff(phi)))

    # Solid angle element dΩ = sin(θ) dθ dφ
    w = np.sin(theta_grid) * dtheta * dphi

    # By default, use all points
    mask = np.ones_like(theta_grid, dtype=bool)

    # If a theta_max is given (in degrees), restrict to theta <= theta_max
    if theta_max_deg is not None:
        theta_max_rad = np.radians(theta_max_deg)
        mask = theta_grid <= theta_max_rad

    # Apply mask to intensities and weights
    I1_m = I1[mask]
    I2_m = I2[mask]
    w_m  = w[mask]

    # Weighted cosine similarity between intensities
    numerator   = np.sum(I1_m * I2_m * w_m)
    denominator = np.sqrt(np.sum(I1_m**2 * w_m) * np.sum(I2_m**2 * w_m))

    overlap = numerator / denominator if denominator != 0 else 0.0
    return float(overlap)


def gaussian_overlap_density_without_pol(
        phi, theta, vals, 
        NA = 0.68, n = 1.0,
        theta_max_deg = None  
):
    """
    Real, non-negative overlap density:
        A(theta,phi) = E_sim * E_gauss * dΩ / sqrt(norm_sim * norm_gauss)
    on the (theta, phi) grid.
    """
    phi   = np.asarray(phi)
    theta = np.asarray(theta)
    vals  = np.asarray(vals)

    assert vals.shape == (len(theta), len(phi)), \
        f"vals has shape {vals.shape}, expected {(len(theta), len(phi))}"

    # Ideal Gaussian amplitude & intensity
    E_gauss = gaussian_far_field_no_polarization(theta, phi, NA, n)
    I_gauss = E_gauss**2

    # Simulated amplitude (from intensity)
    I_sim = vals
    E_sim = np.sqrt(I_sim)

    # Meshgrid
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    dtheta = float(np.mean(np.diff(theta)))
    dphi   = float(np.mean(np.diff(phi)))
    dOmega = np.sin(theta_grid) * dtheta * dphi

    # Region mask in theta
    if theta_max_deg is not None:
        theta_max_rad = np.radians(theta_max_deg)
        mask = theta_grid <= theta_max_rad
    else:
        mask = np.ones_like(theta_grid, dtype=bool)

    norm_sim   = np.sum(I_sim[mask]   * dOmega[mask])
    norm_gauss = np.sum(I_gauss[mask] * dOmega[mask])
    denom = np.sqrt(norm_sim * norm_gauss)

    A_density = np.zeros_like(vals, dtype=float)
    if denom > 0:
        A_density[mask] = (E_sim[mask] * E_gauss[mask] * dOmega[mask]) / denom

    return theta_grid, phi_grid, A_density

def plot_overlap_density(
        phi, theta, vals,
        NA = 0.68, n = 1.0,
        theta_max_deg = 90.0,
        cmap = 'viridis',
        vmin = None, vmax = None,
        show_colorbar = True
):
    """
    Pure polar plot: radius = theta (0–theta_max_deg), angle = phi.
    """

    # Compute overlap density on (theta, phi)
    theta_grid, phi_grid, A_density = gaussian_overlap_density_without_pol(
        phi, theta, vals, NA=NA, n=n, theta_max_deg=theta_max_deg
    )

    # Mask outside desired theta range (for safety)
    theta_max_rad = np.radians(theta_max_deg)
    A_plot = np.array(A_density, copy=True)
    A_plot[theta_grid > theta_max_rad] = np.nan

    # Polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

    # pcolormesh expects angle, radius, Z
    pc = ax.pcolormesh(
        phi_grid, theta_grid, A_plot,
        shading='auto', cmap=cmap, vmin=vmin, vmax=vmax
    )

    # Radius (theta) from 0 to theta_max
    ax.set_ylim(0, theta_max_rad)

    # Radial ticks in degrees (like your far-field plot)
    r_ticks_deg = np.arange(0, theta_max_deg + 10, 10)  # 0,10,...,90
    ax.set_yticks(np.radians(r_ticks_deg))
    ax.set_yticklabels([f'{d:.0f}' for d in r_ticks_deg])

    # (Optional) NA circle overlay
    theta_NA = np.arcsin(NA / n)
    if theta_NA < theta_max_rad:
        ax.plot(np.linspace(0, 2*np.pi, 512),
                np.full(512, theta_NA),
                '--', color='white', linewidth=2)
        ax.text(0.05, theta_NA + np.radians(2),
                f'NA={NA:.2f}, n={n:.1f}',
                color='white', fontsize=10,
                ha='left', va='bottom')

    ax.set_title('Overlap Density: Gaussian Mode', fontsize=13)

    if show_colorbar:
        cbar = fig.colorbar(pc, ax=ax, pad=0.1)
        cbar.set_label('Normalized Overlap Density')

    plt.tight_layout()
    plt.show()
