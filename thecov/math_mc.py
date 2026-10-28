"""This module contains math functions for the covariance calculation.

Same as thecov.math, but with improved sampling of k modes for the Monte Carlo integration over k1
in the window kernels:

- Shell bins are sampled uniformly from the exact lattice modes of each bin (instead of rounding
  continuous points drawn uniformly in r, which over-weighted the inner edge and the lattice
  points near the centre of the shell).
- Modes are drawn without replacement, from a numpy Generator that can be seeded.
- Sampling is stratified in radius and direction (systematic sampling along a path through the
  shell), which keeps every mode equally likely while reducing the Monte Carlo variance.
"""

import numpy as np


def r2c_to_c2c_3d(fourier: np.ndarray) -> np.ndarray:
    """Completes a 3D Fourier array generated using PFFT's r2c method with the elements
    that are omitted due to the Hermitian symmetry of the Fourier transform.

    Parameters
    ----------
    fourier : array_like
        3D Fourier array generated using PFFT's r2c method.

    Returns
    -------
    array_like
        Completed 3D Fourier array.
    """

    # The Fourier transform of a real-valued function is Hermitian, meaning that the values at negative frequencies are the complex conjugates
    # of the values at positive frequencies. The r2c method only stores the non-redundant half of the Fourier array. To complete the array, we
    # need to fill in the values at negative frequencies using the complex conjugates of the values at positive frequencies.
    fourier_c = np.conj(
        fourier[:, :, (-2 if fourier.shape[0] % 2 == 0 else -1) : 0 : -1]
    )

    fourier_c[1:, :, :] = fourier_c[:0:-1, :, :]
    fourier_c[:, 1:, :] = fourier_c[:, :0:-1, :]

    return np.concatenate((fourier, fourier_c), axis=2)


def _get_rng(rng: np.random.Generator | int | None = None) -> np.random.Generator:
    """Returns a numpy random Generator from a seed, an existing Generator or None."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _concatenate_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenates the integer ranges [starts[i], starts[i] + counts[i]) into a single array.

    Equivalent to np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)]), without the loop.
    """
    total = int(np.sum(counts))
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts, counts) + (np.arange(total) - offsets)


def lattice_points_in_shell(
    rmin: float, rmax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerates all integer lattice points (x, y, z) with rmin <= sqrt(x^2 + y^2 + z^2) < rmax.

    The shell is half-open, so consecutive shells [r0, r1), [r1, r2), ... partition the lattice
    without gaps or double counting, which matches the floor(k / dk) binning used for k2 modes
    in SurveyGeometry._compute_window_kernel_row.

    Only the (x, y) plane is gridded; for each (x, y) column the allowed z values are found
    analytically, so the cost scales with the number of points in the shell rather than with
    the volume of the bounding cube.

    Parameters
    ----------
    rmin : float
        Inner radius of the shell (inclusive), in units of the lattice spacing.
    rmax : float
        Outer radius of the shell (exclusive), in units of the lattice spacing.

    Returns
    -------
    ix, iy, iz : np.ndarray of int64
        Coordinates of the lattice points.
    ir : np.ndarray of float64
        Radius of each lattice point.
    """
    rmin = max(float(rmin), 0.0)
    rmax = float(rmax)
    empty = np.zeros(0, dtype=np.int64)
    if rmax <= rmin:
        return empty, empty, empty, np.zeros(0)

    rmin2, rmax2 = rmin**2, rmax**2

    R = int(np.floor(rmax))
    c = np.arange(-R, R + 1, dtype=np.int64)
    ix, iy = np.meshgrid(c, c, indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    s = ix**2 + iy**2

    keep = s < rmax2
    ix, iy, s = ix[keep], iy[keep], s[keep]

    # Largest z >= 0 with s + z^2 < rmax^2 (sqrt followed by integer corrections against round-off)
    zmax = np.floor(np.sqrt(rmax2 - s)).astype(np.int64)
    zmax -= s + zmax**2 >= rmax2
    zmax += s + (zmax + 1) ** 2 < rmax2

    # Smallest z >= 0 with s + z^2 >= rmin^2
    zmin = np.ceil(np.sqrt(np.clip(rmin2 - s, 0, None))).astype(np.int64)
    zmin += s + zmin**2 < rmin2
    zmin -= (zmin > 0) & (s + (zmin - 1) ** 2 >= rmin2)

    # Non-negative branch: z in [zmin, zmax]
    n_pos = np.clip(zmax - zmin + 1, 0, None)
    # Negative branch: z in [-zmax, -max(zmin, 1)], i.e. excluding z = 0, which is counted above
    neg_lo = np.maximum(zmin, 1)
    n_neg = np.clip(zmax - neg_lo + 1, 0, None)

    iz = np.concatenate(
        [_concatenate_ranges(zmin, n_pos), -_concatenate_ranges(neg_lo, n_neg)]
    )
    ix = np.concatenate([np.repeat(ix, n_pos), np.repeat(ix, n_neg)])
    iy = np.concatenate([np.repeat(iy, n_pos), np.repeat(iy, n_neg)])
    ir = np.sqrt(ix**2 + iy**2 + iz**2)

    return ix, iy, iz, ir


def stratified_sample(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    ir: np.ndarray,
    rmin: float,
    rmax: float,
    nsamples: int,
    rng: np.random.Generator | int | None = None,
    stratified: bool = True,
) -> np.ndarray:
    """Selects nsamples distinct lattice points of a shell, each with equal inclusion probability.

    With stratified=True, the points are ordered along a path that sweeps the shell in radial
    sub-shells (equal volume), cos(theta) bands (equal area) and phi cells, with a random order
    inside each cell. Systematic sampling then takes one point every N / nsamples points along
    this path, starting at a random offset. Every point has inclusion probability exactly
    nsamples / N, so the equal-weight average used in the window kernels stays unbiased, while
    the samples are spread evenly over directions and radii, which reduces the Monte Carlo
    variance compared to independent random draws.

    With stratified=False, a simple random sample without replacement is drawn.

    Parameters
    ----------
    ix, iy, iz, ir : np.ndarray
        Lattice points of the shell, as returned by lattice_points_in_shell.
    rmin, rmax : float
        Inner and outer radii of the shell.
    nsamples : int
        Number of points to select. If nsamples >= N, all points are returned.
    rng : np.random.Generator, int or None
        Random number generator or seed.
    stratified : bool
        Whether to use stratified (systematic) sampling. Default is True.

    Returns
    -------
    np.ndarray
        Integer indices of the selected points, in random order.
    """
    rng = _get_rng(rng)
    N = len(ir)
    nsamples = int(nsamples)

    if nsamples >= N:
        return rng.permutation(N)

    if not stratified:
        return rng.choice(N, size=nsamples, replace=False)

    # Number of strata. Radial strata are only useful when the shell spans several lattice units;
    # the remaining strata go to directions, split into equal-area cos(theta) bands and phi cells.
    n_r = max(1, min(int(rmax - rmin), int(round(nsamples ** (1 / 3)))))
    n_dir = max(1, nsamples // n_r)
    n_mu = max(1, int(round(np.sqrt(n_dir / 2))))
    n_phi = max(1, int(round(n_dir / n_mu)))

    # Radial stratum, using equal-volume sub-shells
    r3 = (ir**3 - rmin**3) / (rmax**3 - rmin**3)
    i_r = np.clip((r3 * n_r).astype(np.int64), 0, n_r - 1)

    # cos(theta) band (equal area on the sphere). The origin, if present, goes to the middle band.
    safe_r = np.where(ir > 0, ir, 1.0)
    mu = np.where(ir > 0, iz / safe_r, 0.0)
    i_mu = np.clip(((mu + 1) / 2 * n_mu).astype(np.int64), 0, n_mu - 1)

    # phi cell
    phi = np.mod(np.arctan2(iy, ix), 2 * np.pi)
    i_phi = np.clip((phi / (2 * np.pi) * n_phi).astype(np.int64), 0, n_phi - 1)

    # Serpentine ordering, so that consecutive cells along the path are neighbours on the sphere
    i_mu = np.where(i_r % 2 == 1, n_mu - 1 - i_mu, i_mu)
    i_phi = np.where((i_r * n_mu + i_mu) % 2 == 1, n_phi - 1 - i_phi, i_phi)
    cell = (i_r * n_mu + i_mu) * n_phi + i_phi

    # Sort by cell, with a random order inside each cell
    order = np.lexsort((rng.random(N), cell))

    # Systematic sampling along the path: one point every N / nsamples points, random start
    step = N / nsamples
    picks = np.floor(rng.uniform(0, step) + step * np.arange(nsamples)).astype(np.int64)
    picks = np.clip(picks, 0, N - 1)

    # Random final order, so that splitting the samples in chunks across workers
    # (SurveyGeometry.compute_window_kernels) gives chunks that each cover the whole shell
    return rng.permutation(order[picks])


def sample_from_shell(
    rmin: float,
    rmax: float,
    discrete: bool = True,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float, float, float]:
    """Sample a point uniformly from a spherical shell.

    Parameters
    ----------
    rmin : float
        Minimum radius of the shell.
    rmax : float
        Maximum radius of the shell.
    discrete : bool, optional
        If True, the sampled point is drawn uniformly from the integer lattice points with
        rmin <= r < rmax. If False, a continuous point is drawn uniformly in volume.
        Default is True.
    rng : np.random.Generator, int or None
        Random number generator or seed.

    Returns
    -------
    x,y,z,r : float
        Coordinates of the sampled point.
    """
    rng = _get_rng(rng)

    if discrete:
        # Rejection sampling from the bounding cube: every lattice point of the shell is
        # accepted with the same probability, so the result is uniform over the shell's lattice points
        R = int(np.floor(rmax))
        while True:
            x, y, z = (int(v) for v in rng.integers(-R, R + 1, size=3))
            r = np.sqrt(x**2 + y**2 + z**2)
            if rmin <= r < rmax:
                return x, y, z, r

    # Uniform in volume: r^3 is uniform between rmin^3 and rmax^3. See https://mathworld.wolfram.com/SpherePointPicking.html
    r = (rmin**3 + (rmax**3 - rmin**3) * rng.random()) ** (1 / 3)
    theta = 2 * np.pi * rng.random()
    phi = np.arccos(1 - 2 * rng.random())

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return x, y, z, r


def sample_from_cube(
    rmin: float,
    rmax: float,
    dr: float,
    max_modes: int | float = np.inf,
    rng: np.random.Generator | int | None = None,
    stratified: bool = True,
) -> tuple[list[np.ndarray], list[int]]:
    """Sample lattice points within consecutive spherical shells [rmin + i dr, rmin + (i + 1) dr).

    All lattice points of each shell are enumerated. If a shell has more than max_modes points,
    max_modes distinct points are selected with stratified_sample.

    Parameters
    ----------
    rmin : float
        Minimum radius of the first shell.
    rmax : float
        Maximum radius of the last shell.
    dr : float
        Width of the radial bins.
    max_modes : int | float, optional
        Maximum number of modes to sample. Default is infinity.
    rng : np.random.Generator, int or None
        Random number generator or seed.
    stratified : bool
        Whether to use stratified sampling when subsampling a shell. Default is True.

    Returns
    -------
    modes : list of np.ndarray
        Sampled modes, each with shape (n, 4) and columns (x, y, z, r).
    Nmodes : list of int
        Number of lattice modes in each bin.
    """
    rng = _get_rng(rng)

    imax = (rmax - rmin) / dr
    imax = int(np.round(imax)) if np.allclose(np.round(imax), imax) else int(imax)

    modes = []
    Nmodes = []

    for i in range(imax):
        lo, hi = rmin + i * dr, rmin + (i + 1) * dr
        ix, iy, iz, ir = lattice_points_in_shell(lo, hi)
        N = len(ir)
        if N > max_modes:
            idx = stratified_sample(
                ix, iy, iz, ir, lo, hi, int(max_modes), rng=rng, stratified=stratified
            )
        else:
            idx = np.arange(N)
        modes.append(np.array([ix[idx], iy[idx], iz[idx], ir[idx]], dtype=np.float64).T)
        Nmodes.append(N)

    return modes, Nmodes


def sample_kmodes(
    kmin: float | None,
    kmax: float | None,
    dk: float | None,
    boxsize: float,
    max_modes: int = 1000,
    k_shell_approx: float = 0.05,
    rng: np.random.Generator | int | None = None,
    stratified: bool = True,
    exact_nmodes: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Sample Fourier modes in each k bin, to be used in the Monte Carlo integration over k1.

    Every bin, below and above k_shell_approx, is sampled from the exact set of lattice modes in
    the bin, without replacement and (by default) stratified in radius and direction.
    k_shell_approx now only controls how the number of modes per bin is computed: exact
    lattice counts below it, and the continuous shell volume above it (as in thecov.math),
    unless exact_nmodes=True, in which case exact counts are used for all bins.

    Parameters
    ----------
    kmin, kmax, dk : float
        Edges and width of the k bins.
    boxsize : float
        Size of the box, which sets the fundamental frequency.
    max_modes : int
        Maximum number of modes sampled per bin.
    k_shell_approx : float
        Wavenumber above which the number of modes per bin uses the continuous shell volume.
    rng : np.random.Generator, int or None
        Random number generator or seed. Pass an int for reproducible kernels.
    stratified : bool
        Whether to use stratified sampling. Default is True.
    exact_nmodes : bool
        Whether to use exact lattice counts for the number of modes in all bins. Default is False.

    Returns
    -------
    modes : list of np.ndarray
        Sampled modes in each bin, each with shape (n, 4) and columns (x, y, z, r) in units of kfun.
    Nmodes : np.ndarray
        Number of modes in each bin.
    """
    import logging

    logger = logging.getLogger("SampleModes")

    assert kmin is not None and kmax is not None and dk is not None, (
        "kmin, kmax, and dk must be specified."
    )

    rng = _get_rng(rng)

    # Wavelength where spherical shell approximation kicks in
    k_shell = max((k_shell_approx - kmin) // dk * dk + kmin, kmin)

    # The fundamental frequency of the box
    kfun = 2 * np.pi / boxsize

    # Bins from kmin to k_shell, with exact mode counts
    cube_modes, cube_nmodes = sample_from_cube(
        kmin / kfun,
        k_shell / kfun,
        dk / kfun,
        max_modes=max_modes,
        rng=rng,
        stratified=stratified,
    )

    # Bins from k_shell to kmax, sampled from the exact lattice modes of each shell
    kedges_shell = np.arange(k_shell, kmax + dk / 2, dk)
    shell_modes = []
    shell_counts = []
    for lo, hi in zip(kedges_shell[:-1] / kfun, kedges_shell[1:] / kfun):
        ix, iy, iz, ir = lattice_points_in_shell(lo, hi)
        idx = stratified_sample(
            ix, iy, iz, ir, lo, hi, max_modes, rng=rng, stratified=stratified
        )
        shell_modes.append(
            np.array([ix[idx], iy[idx], iz[idx], ir[idx]], dtype=np.float64).T
        )
        shell_counts.append(len(ir))

    if exact_nmodes:
        shell_nmodes = shell_counts
    else:
        shell_nmodes = list(
            np.atleast_1d(
                nmodes(
                    boxsize**3,
                    np.float64(kedges_shell[:-1]),
                    np.float64(kedges_shell[1:]),
                )
            )
        )

    logger.info(
        f"Sampled {len(cube_modes)} bins with exact mode counts and {len(shell_modes)} bins "
        f"with {'exact' if exact_nmodes else 'continuous'} mode counts "
        f"({'stratified' if stratified else 'simple random'} sampling, without replacement)."
    )

    return cube_modes + shell_modes, np.array(cube_nmodes + shell_nmodes)


def nmodes(
    volume: float, kmin: float | np.ndarray, kmax: float | np.ndarray
) -> float | np.ndarray:
    """Compute the number of modes in a given shell.

    Parameters
    ----------
    volume : float
        Volume of the survey.
    kmin : float
        Minimum k of the shell.
    kmax : float
        Maximum k of the shell.

    Returns
    -------
    float
        Number of modes.
    """
    return volume / 3.0 / (2 * np.pi**2) * (kmax**3 - kmin**3)


def cov2cor(covariance: np.ndarray) -> np.ndarray:
    """Compute the correlation matrix from the covariance matrix.

    Parameters
    ----------
    covariance : array_like
        Covariance matrix.

    Returns
    -------
    array_like
        Correlation matrix."""
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def fgrowth(Omega_m: float, z: float) -> float:
    """Estimates the growth rate at redshift z.

    Parameters
    ----------
    Omega_m : float
        Matter density parameter.
    z : float
        Redshift.

    Returns
    -------
    float
        Growth rate.
    """
    from scipy.special import hyp2f1

    return 1.0 + 6 * (Omega_m - 1) * hyp2f1(
        4 / 3.0, 2, 17 / 6.0, (1 - 1 / Omega_m) / (1 + z) ** 3
    ) / (
        11
        * Omega_m
        * (1 + z) ** 3
        * hyp2f1(1 / 3.0, 1, 11 / 6.0, (1 - 1 / Omega_m) / (1 + z) ** 3)
    )


def wedges_to_multipoles(
    corr: np.ndarray, muedges: np.ndarray, ells=(0, 2, 4)
) -> np.ndarray:
    from scipy import special

    corr_ell = []
    for ell in ells:
        poly = special.legendre(ell).integ()(muedges)
        legendre = (2 * ell + 1) * (poly[1:] - poly[:-1])
        corr_ell.append(np.sum(corr * legendre, axis=-1) / (muedges[-1] - muedges[0]))
    return np.array(corr_ell)


def legendre(ell: int) -> np.vectorize:
    if ell == 0:
        legendre = lambda mu: 1
    elif ell == 2:
        legendre = lambda mu: (3 * mu**2 - 1) / 2
    elif ell == 4:
        legendre = lambda mu: (35 * mu**4 - 30 * mu**2 + 3) / 8
    else:
        raise ValueError("Only ell=0,2,4 are supported.")

    return np.vectorize(legendre)


def get_real_Ylm(ell: int, m: int, modules=None):
    """
    Return a function that computes the real spherical harmonic of order (ell, m).
    Copied from pypower.
    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py.

    Note
    ----
    Faster evaluation will be achieved if sympy and numexpr are available.
    Else, fallback to numpy and scipy's functions.

    Parameters
    ----------
    ell : int
        The degree of the harmonic.

    m : int
        The order of the harmonic; abs(m) <= ell.

    modules : str, default=None
        If 'sympy', use sympy + numexpr to speed up calculation.
        If 'scipy', use scipy.
        If ``None``, defaults to sympy if installed, else scipy.

    Returns
    -------
    Ylm : callable
        A function that takes 3 arguments: (xhat, yhat, zhat)
        unit-normalized Cartesian coordinates and returns the
        specified Ylm.

    References
    ----------
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    """
    # Make sure ell, m are integers
    ell = int(ell)
    m = int(m)

    # Normalization of Ylms
    amp = np.sqrt((2 * ell + 1) / (4 * np.pi))
    if m != 0:
        fac = 1
        for n in range(ell - abs(m) + 1, ell + abs(m) + 1):
            fac *= n  # (ell + |m|)!/(ell - |m|)!
        amp *= np.sqrt(2.0 / fac)

    sp = None

    if modules is None:
        try:
            import sympy as sp
        except ImportError:
            pass

    elif "sympy" in modules:
        import sympy as sp

    elif "scipy" not in modules:
        raise ValueError('modules must be either ["sympy", "scipy", None]')

    # sympy is not installed, fallback to scipy
    if sp is None:
        import scipy.special

        def Ylm(xhat, yhat, zhat):
            # The cos(theta) dependence encoded by the associated Legendre polynomial
            toret = amp * (-1) ** m * scipy.special.lpmv(abs(m), ell, zhat)
            # The phi dependence
            phi = np.arctan2(yhat, xhat)
            if m < 0:
                toret *= np.sin(abs(m) * phi)
            else:
                toret *= np.cos(abs(m) * phi)
            return toret

        # Attach some meta-data
        Ylm.l = ell  # type: ignore
        Ylm.m = m  # type: ignore
        return Ylm

    # The relevant cartesian and spherical symbols
    # Using intermediate variable r helps sympy simplify expressions
    x, y, z, r = sp.symbols("x y z r", real=True, positive=True)
    xhat, yhat, zhat = sp.symbols("xhat yhat zhat", real=True, positive=True)
    phi, theta = sp.symbols("phi theta")
    defs = [
        (sp.sin(phi), y / sp.sqrt(x**2 + y**2)),
        (sp.cos(phi), x / sp.sqrt(x**2 + y**2)),
        (sp.cos(theta), z / sp.sqrt(x**2 + y**2 + z**2)),
    ]

    # The cos(theta) dependence encoded by the associated Legendre polynomial
    expr = (-1) ** m * sp.assoc_legendre(ell, abs(m), sp.cos(theta))

    # The phi dependence
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m) * phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m * phi))

    # Simplify
    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x / r, xhat), (y / r, yhat), (z / r, zhat)])

    try:
        import numexpr
    except ImportError:
        numexpr = None
    Ylm = sp.lambdify(
        (xhat, yhat, zhat),
        expr,
        modules="numexpr" if numexpr is not None else ["scipy", "numpy"],
    )

    # Attach some meta-data
    Ylm.expr = expr  # type: ignore
    Ylm.l = ell  # type: ignore
    Ylm.m = m  # type: ignore
    return Ylm
