import thecov.geometry
import thecov.covariance

# Libraries to handle basic cosmology and catalog manipulations
from mpytools import Catalog
import mockfactory.utils
import cosmoprimo
import numpy as np

from scipy.interpolate import interp1d

# Define fiducial cosmology used in calculations
cosmo = cosmoprimo.fiducial.DESI()

# Load random catalog
randoms = Catalog.read("/home/ylai1998/allcov/6dFGS_random.fits")

# Any catalog filtering/manipulations should go here

# The characteristic amplitude of the power spectrum
P0_characteristic = 1e4
# Whether RA and Dec are in degrees.
degree = False
# The Nyquist wavelength of the window FFTs should be at least as large as the largest k-bin used in covariance calculations.
kmax_window = 0.04
# Multiply the box size inferred from the catalog by this factor to get the box size used in window FFTs.
# Larger boxpad yields smaller k-fundamental.
boxpad = 2.0
# The ratio of the number of galaxies to the number of randoms. Should be used in covariance calculations to properly account
# for shot noise.
alpha = 1.0
# Maixmum number of samples used in the Monte Carlo integration of the trispectrum covariance. Should be set to a large number for
# accurate results, but can be reduced for faster tests.
kmodes_sampled = 5000


# Should define FKP weights column with this name
randoms["WEIGHT_FKP"] = 1.0 / (1.0 + P0_characteristic * randoms["NZ"])  # type: ignore  # FKP weights are optional

# Convert sky coordinates to cartesian using fiducial cosmology
randoms["POSITION"] = mockfactory.utils.sky_to_cartesian(
    cosmo.comoving_radial_distance(randoms["Z"]),
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
    kmodes_sampled=kmodes_sampled,  # max N samples used in integration
)

kmin = 0.01
kmax = 0.30
dk = 0.005

# kmin, kmax, dk = 0.0, 0.5, 0.005

gaussian = thecov.covariance.GaussianCovariance(geometry)
gaussian.set_kbins(kmin, kmax, dk)

powerspectra = np.load("/home/ylai1998/allcov/PS_MP_NGC_0_6dFGS_best_model.npy")
length = int(len(powerspectra) / 3)
P0 = powerspectra[:length]
P2 = powerspectra[length : 2 * length]
P4 = powerspectra[2 * length :]

kmode = np.load("/home/ylai1998/allcov/kmode_NGC_0_6dFGS.npy")
kmode = kmode[:length]

P0 = interp1d(kmode, P0, kind="cubic", bounds_error=True, fill_value=0.0)(gaussian.kmid)
P2 = interp1d(kmode, P2, kind="cubic", bounds_error=True, fill_value=0.0)(gaussian.kmid)
P4 = interp1d(kmode, P4, kind="cubic", bounds_error=True, fill_value=0.0)(gaussian.kmid)

# Load input power spectra (P0, P2, P4) for the Gaussian covariance

gaussian.set_galaxy_pk_multipole(P0, 0, has_shotnoise=False)
gaussian.set_galaxy_pk_multipole(P2, 2)
gaussian.set_galaxy_pk_multipole(P4, 4)

gaussian.compute_covariance()

# raise ValueError(
#     "This is not a test. Just a script to check that the code runs without errors. Should be deleted or converted to a real test."
# )

# Galaxy bias b1 and effective redshift zeff
b1, zeff = 2.0, 0.5

t0 = thecov.covariance.RegularTrispectrumCovariance(geometry)
t0.set_kbins(kmin, kmax, dk)

plin = cosmo.get_fourier()

t0.set_linear_matter_pk(np.vectorize(lambda k: plin.pk_kz(k, zeff)))

# Other bias parameters will be automatically determined
# assuming local lagrangian approximation if not given
t0.set_params(fgrowth=cosmo.growth_rate(zeff), b1=b1)
t0.compute_covariance()

# Creating a new geometry object with finer grid for SSC calcs.
# Larger boxpad yields smaller k-fundamental.
geometry_ssc = thecov.geometry.SurveyGeometry(
    randoms, kmax_window=0.1, boxpad=2.0, alpha=0.1
)
ssc = thecov.covariance.SuperSampleCovariance(geometry_ssc)
ssc.set_kbins(kmin, kmax, dk)
ssc.set_linear_matter_pk(np.vectorize(lambda k: plin.pk_kz(k, zeff)))
ssc.set_params(fgrowth=cosmo.growth_rate(zeff), b1=b1)
ssc.compute_covariance()

covariance = gaussian + t0 + ssc
