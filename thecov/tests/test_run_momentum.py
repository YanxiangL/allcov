import thecov.geometry
import thecov.covariance

# Libraries to handle basic cosmology and catalog manipulations
from mpytools import Catalog
import mockfactory.utils
import cosmoprimo
import numpy as np

# Script to check that the Gaussian covariance of the momentum auto-power spectrum (pp-pp, eq. 2.33)
# runs without errors. Should be converted to a real test or deleted in the future.

# Define fiducial cosmology used in calculations
cosmo = cosmoprimo.fiducial.DESI()

# Load random catalog of the peculiar velocity sample
randoms = Catalog.read(
    "/home/ylai1998/allcov/PS_code/data/PV_clustering_random20_v5_v13.fits"
)

# Any catalog filtering/manipulations should go here

# The characteristic amplitude of the momentum power spectrum (FKPv in PS_code/PSestFun.py)
Pp_characteristic = 5e9
# Fractional distance uncertainty of the Fundamental Plane / Tully-Fisher relations
sigma_dist = 0.2
# Nonlinear velocity dispersion in km/s
sigma_v_nl = 300.0
# Hubble constant in km/s/(Mpc/h), so that H0 * D is in km/s for D in Mpc/h
H0 = 100.0
# Whether RA and Dec are in degrees.
degree = True
# The Nyquist wavelength of the window FFTs should be at least as large as the largest k-bin used in covariance calculations.
kmax_window = 0.04
# Multiply the box size inferred from the catalog by this factor to get the box size used in window FFTs.
# Larger boxpad yields smaller k-fundamental.
boxpad = 2.0
# The ratio of the number of PV galaxies to the number of randoms. The catalog has 20x randoms.
alpha = 1.0 / 20
# Maixmum number of samples used in the Monte Carlo integration of the trispectrum covariance. Should be set to a large number for
# accurate results, but can be reduced for faster tests.
kmodes_sampled = 5000

# Number density of the peculiar velocity sample
randoms["NZ"] = randoms["NPV"]

# Comoving distance in Mpc/h
distance = cosmo.comoving_radial_distance(randoms["ZCMB"])

# Velocity variance <u^2(x)> in (km/s)^2: measurement uncertainty plus nonlinear velocity dispersion
randoms["VAR_U"] = (sigma_dist * H0 * distance) ** 2 + sigma_v_nl**2

# FKP weights for the momentum field (eq. 2.2)
randoms["WEIGHT_FKP"] = 1.0 / (randoms["VAR_U"] + Pp_characteristic * randoms["NZ"])  # type: ignore

# Convert sky coordinates to cartesian using fiducial cosmology
randoms["POSITION"] = mockfactory.utils.sky_to_cartesian(
    distance,
    randoms["RA"],
    randoms["DEC"],
    degree=degree,
)

# Create geometry object to be used in covariance calculation
geometry = thecov.geometry.SurveyGeometry(
    randoms,
    kmax_window=kmax_window,  # Nyquist wavelength of window FFTs
    boxpad=boxpad,  # multiplies the box size inferred from catalog
    alpha=alpha,  # N_galaxies / N_randoms
    velocity=True,  # momentum field: W12 includes VAR_U and no (1 + alpha) shot-noise factors
    kmodes_sampled=kmodes_sampled,  # max N samples used in integration
)

# Same bins as mean_mom_PS.npz (centres 0.0045, 0.0135, ...)
kmin = 0.0
kmax = 0.297
dk = 0.009

gaussian = thecov.covariance.GaussianCovariance(geometry)
gaussian.set_kbins(kmin, kmax, dk)

# Load input momentum power spectrum multipoles (P0, P2, P4) for the Gaussian covariance
powerspectra = np.load("/home/ylai1998/allcov/PS_code/data/mean_mom_PS.npz")
nbins = len(gaussian.kmid)
assert np.allclose(powerspectra["k"][:nbins], gaussian.kmid), (
    "k bins of the covariance do not match the input power spectrum."
)
P0 = powerspectra["P0"][:nbins]
P2 = powerspectra["P2"][:nbins]
P4 = powerspectra["P4"][:nbins]

# PSestFun subtracts the shot-noise (Pvnoise) from the momentum monopole
gaussian.set_galaxy_pk_multipole(P0, 0, has_shotnoise=False)
gaussian.set_galaxy_pk_multipole(P2, 2)
gaussian.set_galaxy_pk_multipole(P4, 4)

print(f"Momentum shot-noise = {gaussian.shotnoise:.3e} (km/s)^2 (Mpc/h)^3")

gaussian.compute_covariance()

print(f"Covariance shape: {gaussian.cov.shape}")
print(f"Symmetric: {np.allclose(gaussian.cov, gaussian.cov.T)}")
print(f"Positive definite: {(gaussian.eigvals > 0).all()}")
