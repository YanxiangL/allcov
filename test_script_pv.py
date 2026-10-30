import thecov.geometry
import thecov.covariance

# Libraries to handle basic cosmology and catalog manipulations
from mpytools import Catalog
import mockfactory.utils
import cosmoprimo
import numpy as np

from scipy.interpolate import interp1d
from pypower import PowerSpectrumMultipoles

# Define fiducial cosmology used in calculations
cosmo = cosmoprimo.fiducial.DESI()

# Load random catalog
randoms = Catalog.read("/pscratch/sd/l/lgong/pv_cov/updated_desiy1_random_catalog_abacus_mock.fits")
#randoms = Catalog.read('/pscratch/sd/l/lgong/pv_cov/LRG_NGC_0_clustering_cut_z0.4-0.6.ran.fits')
# Any catalog filtering/manipulations should go here

# The characteristic amplitude of the power spectrum
P0_characteristic = 1e10
#P0_characteristic = 1e4
# Whether RA and Dec are in degrees.
degree = True
# The Nyquist wavelength of the window FFTs should be at least as large as the largest k-bin used in covariance calculations.
kmax_window = 0.04
# Multiply the box size inferred from the catalog by this factor to get the box size used in window FFTs.
# Larger boxpad yields smaller k-fundamental.
boxpad = 2.0
# The ratio of the number of galaxies to the number of randoms. Should be used in covariance calculations to properly account
# for shot noise.
alpha = 0.05
#alpha = 0.156
# Maixmum number of samples used in the Monte Carlo integration of the trispectrum covariance. Should be set to a large number for
# accurate results, but can be reduced for faster tests.
kmodes_sampled = 5000


# Should define FKP weights column with this name
# randoms["WEIGHT_FKP"] = 1.0 / (1.0 + P0_characteristic * randoms["NZ"])  # type: ignore  # FKP weights are optional
#randoms["WEIGHT_FKP"] = np.ones_like(randoms["WEIGHT_FKP"])

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
    resume_file="/pscratch/sd/l/lgong/pv_cov/resume_BGS_pv_DESIY1_P0_P2_P4_Gaussian_wo_SN.pkl"
)


"""
geometry = thecov.geometry.BoxGeometry(
    volume = 3.56e7,
    nbar = 0.0027597553081437727
)
"""
kmin = 0.01
kmax = 0.42 
dk = 0.005

# kmin, kmax, dk = 0.0, 0.5, 0.005

gaussian = thecov.covariance.GaussianCovariance(geometry)
gaussian.set_kbins(kmin, kmax, dk)

"""
powerspectra = np.load('/pscratch/sd/l/lgong/pv_cov/pk_model_output.npz')
k_mom = powerspectra['k_mom']
pk_mom = powerspectra['pk_mom']

# DESI DR1 mock average peculiar velocity power spectrum
kmode = k_mom[:1000]
P0 = pk_mom[:1000]
P2 = pk_mom[1000:2000]
P4 = pk_mom[2000:3000]
"""

# Yan's analytically predicted DESI DR1 peculiar velocity power spectrum
pv = np.load('/pscratch/sd/l/lgong/pv_cov/Pkmulti_mom.npy')
length = 1000
P0 = pv[1, :length]
P2 = pv[1, length:2*length]
P4 = pv[1, 2*length:]

kmode = pv[0]
kmode = kmode[:length]


'''
# DESI real data galaxy clustering power spectrum
pk = PowerSpectrumMultipoles.load('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/pkpoles_LRG_NGC_z0.4-0.6.npy')
kmode = pk.k

# Extract your multipoles (automatically normalized, real-valued, and shot-noise corrected!)
P0 = pk(ell=0)
P2 = pk(ell=2)
P4 = pk(ell=4)

# You can even extract complex cross-poles or format it beautifully
print(f"Available multipoles: {pk.ells}")
print(f"Shot noise applied: {pk.shotnoise}")
'''

P0 = interp1d(kmode, P0, kind="cubic", bounds_error=True, fill_value=0.0)(gaussian.kmid)
P2 = interp1d(kmode, P2, kind="cubic", bounds_error=True, fill_value=0.0)(gaussian.kmid)
P4 = interp1d(kmode, P4, kind="cubic", bounds_error=True, fill_value=0.0)(gaussian.kmid)

# Load input power spectra (P0, P2, P4) for the Gaussian covariance

gaussian.set_galaxy_pk_multipole(P0, 0, has_shotnoise=False)
gaussian.set_galaxy_pk_multipole(P2, 2, has_shotnoise=False)
gaussian.set_galaxy_pk_multipole(P4, 4, has_shotnoise=False)

gaussian.compute_covariance()

# raise ValueError(
#     "This is not a test. Just a script to check that the code runs without errors. Should be deleted or converted to a real test."
# )
"""
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
"""

covariance = gaussian

np.save('/pscratch/sd/l/lgong/pv_cov/BGS_DESIY1_P0_P2_P4_Gaussian_cov_wo_shotshoise_analytic_pv.npy', covariance.cov)
#np.save('/pscratch/sd/l/lgong/pv_cov/LRG_NGC_clustering_DESIY1_P0_P2_Gaussian_cov_no_shotshoise.npy', covariance.cov)