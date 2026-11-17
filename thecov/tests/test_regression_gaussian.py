"""Regression test for the Gaussian covariance of a survey geometry.

What it checks
--------------
That the window kernels, the Gaussian covariance and the shot-noise of the galaxy (C_gggg) and
momentum (C_pppp) auto-power spectra stay the same when the code is changed. It does not check
that the covariance is physically correct: the setup is a small toy survey, chosen so that the
test is fast. The reference outputs were computed with the original (CovaPT) window kernel code,
before thecov/kernels.py existed. Every output is checked with both kernel backends of
SurveyGeometry: "covapt" (the original code) and "generic" (thecov.kernels), which must agree
with the same reference. The reference file is always regenerated with "covapt".

How it works
------------
1. Build a toy random catalog (uniform in a spherical shell, see make_randoms).
2. Run SurveyGeometry + GaussianCovariance on it, once for galaxies (velocity=False) and once for
   the momentum field (velocity=True), with toy power spectrum multipoles (see compute_outputs).
3. Compare the outputs with the reference file thecov/tests/reference/gaussian_reference.npz.

Everything is deterministic: the randoms use a fixed seed, the k1 modes are sampled with a fixed
seed in SurveyGeometry.compute_window_kernels (math_mc.sample_kmodes(..., rng=42)), and the
window kernels are computed with one thread (see compute_outputs). So any difference beyond
floating-point round-off means that the results of the code have changed.

Running it
----------
    pytest thecov/tests/test_regression_gaussian.py     # run the test (about a minute)
    python thecov/tests/test_regression_gaussian.py     # regenerate the reference file

Only regenerate the reference file when a change of the results is intended (e.g. a bug fix),
and say so in the commit message, since the test can no longer catch changes made before it.
"""

import os

import numpy as np
import pytest

import thecov.covariance
import thecov.geometry

# Reference outputs, saved with np.savez by running this file as a script (see the bottom).
# Keys are "<block>_<output>", e.g. "gggg_WinKernel" or "pppp_cov".
REFERENCE_FILE = os.path.join(
    os.path.dirname(__file__), "reference", "gaussian_reference.npz"
)

# ---------------------------------------------------------------------------------------------
# Toy setup. Kept small so that the test runs in about a minute; changing any of these values
# changes the outputs, so the reference file has to be regenerated afterwards.
# ---------------------------------------------------------------------------------------------

# Number of random points
NRANDOMS = 50_000
# Inner and outer radius of the spherical shell of the toy survey, in Mpc/h
RMIN, RMAX = 200.0, 400.0
# Number of mesh points per dimension of the window FFTs (sets the cost of every window kernel)
NMESH = 32
# k bins of the covariance, in h/Mpc. kmax is above k_shell_approx = 0.1 in sample_kmodes, so both
# the exact-count and shell-volume bins are tested. (kmax < 0.1 also fails because of a bug in
# math_mc.sample_kmodes, which then returns more bins than requested.)
KMIN, KMAX, DK = 0.0, 0.12, 0.012
# Number of k1 modes sampled per bin in the Monte Carlo integration of the window kernels.
# Much smaller than in real runs (~5000): the covariance is noisy (not symmetric, not positive
# definite), but that does not matter for a regression test.
KMODES_SAMPLED = 20
# Ratio of the number of data points to the number of randoms
ALPHA = 0.1

# Characteristic amplitude of the power spectrum, used in the FKP weights and the toy multipoles.
# The momentum power spectrum is in (km/s)^2 (Mpc/h)^3, hence much larger.
P_CHARACTERISTIC = {"gggg": 1e4, "pppp": 5e9}


def make_randoms(velocity: bool, seed: int = 0) -> dict:
    """Toy random catalog, uniform in a spherical shell between RMIN and RMAX.

    Parameters
    ----------
    velocity : bool
        If True, the catalog is for the momentum field: it has the velocity variance column
        VAR_U and momentum FKP weights. If False, it has galaxy FKP weights.
    seed : int
        Seed of the random number generator, so that the catalog is the same in every run.

    Returns
    -------
    dict
        Columns POSITION, NZ, WEIGHT, WEIGHT_FKP (and VAR_U if velocity=True), as expected by
        SurveyGeometry.
    """
    rng = np.random.default_rng(seed)

    # Positions uniform in volume: r^3 is uniform between RMIN^3 and RMAX^3, and the direction
    # is uniform on the sphere (mu = cos(theta) uniform in [-1, 1], phi uniform in [0, 2 pi]).
    r = (RMIN**3 + (RMAX**3 - RMIN**3) * rng.random(NRANDOMS)) ** (1 / 3)
    mu = 2 * rng.random(NRANDOMS) - 1
    phi = 2 * np.pi * rng.random(NRANDOMS)
    sin_theta = np.sqrt(1 - mu**2)
    position = r[:, None] * np.column_stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), mu]
    )

    # Number density in (h/Mpc)^3, decreasing with distance (from 3e-4 at RMIN to 1.5e-4 at
    # RMAX), so that the window and the FKP weights depend on position as in a real survey.
    nz = 3e-4 * (1 - 0.5 * (r - RMIN) / (RMAX - RMIN))

    randoms = {
        "POSITION": position,
        "NZ": nz,
        "WEIGHT": np.ones(NRANDOMS),
    }
    if velocity:
        # Velocity variance <u^2> in (km/s)^2: 20% distance error (sigma * H0 * D, with
        # H0 = 100 km/s/(Mpc/h)) plus the 300 km/s nonlinear velocity dispersion.
        randoms["VAR_U"] = (0.2 * 100.0 * r) ** 2 + 300.0**2
        # Momentum FKP weights, 1 / (<u^2> + nbar * P_p)
        randoms["WEIGHT_FKP"] = 1 / (randoms["VAR_U"] + P_CHARACTERISTIC["pppp"] * nz)
    else:
        # Galaxy FKP weights, 1 / (1 + nbar * P_g)
        randoms["WEIGHT_FKP"] = 1 / (1 + P_CHARACTERISTIC["gggg"] * nz)

    return randoms


def compute_outputs(velocity: bool, kernel_backend: str = "covapt") -> dict:
    """Window kernels, Gaussian covariance and shot-noise for the toy setup.

    Parameters
    ----------
    velocity : bool
        If True, computes C_pppp (momentum field). If False, computes C_gggg (galaxies).
    kernel_backend : str
        Code that computes the window kernels: "covapt" (the original code, used to make the
        reference file) or "generic" (thecov.kernels). Both must reproduce the reference.

    Returns
    -------
    dict
        WinKernel : window kernels, shape (kbins, 2 * delta_k_max + 1, 15, 6)
        cov : Gaussian covariance of the (P0, P2, P4) multipoles, shape (3 * kbins, 3 * kbins)
        shotnoise : shot-noise of the power spectrum (0-d array)
    """
    geometry = thecov.geometry.SurveyGeometry(
        make_randoms(velocity),
        nmesh=NMESH,
        alpha=ALPHA,
        velocity=velocity,
        kernel_backend=kernel_backend,
        kmodes_sampled=KMODES_SAMPLED,
        # The window kernels are a sum over the chunks of each worker, so the
        # number of threads changes the floating-point summation order.
        nthreads=1,
    )

    gaussian = thecov.covariance.GaussianCovariance(geometry)
    gaussian.set_kbins(KMIN, KMAX, DK)

    # Toy power spectrum multipoles: a smooth, positive monopole, with P2 and P4 proportional to it.
    # Only their values matter here, not whether they are realistic. The monopole is set without
    # shot-noise, so no shot-noise is subtracted from it.
    k = gaussian.kmid
    P0 = P_CHARACTERISTIC["pppp" if velocity else "gggg"] * (0.02 / (k + 0.01))
    gaussian.set_galaxy_pk_multipole(P0, 0, has_shotnoise=False)
    gaussian.set_galaxy_pk_multipole(0.5 * P0, 2)
    gaussian.set_galaxy_pk_multipole(0.1 * P0, 4)

    # Computes the window kernels (the slow part) and then the covariance
    gaussian.compute_covariance()

    return {
        # Checked directly because it is more sensitive than the covariance: it includes the
        # shot-noise kernels (W12, with <u^2> for the momentum field) term by term.
        "WinKernel": geometry.WinKernel,
        "cov": gaussian.cov,
        # Checks that the (1 + alpha) factor is dropped and <u^2> included for the momentum field
        "shotnoise": np.array(gaussian.shotnoise),
    }


def compute_all_outputs(kernel_backend: str = "covapt") -> dict:
    """Outputs of compute_outputs for both C_gggg and C_pppp, with keys "<block>_<output>"."""
    outputs = {}
    for name, velocity in [("gggg", False), ("pppp", True)]:
        for key, value in compute_outputs(velocity, kernel_backend).items():
            outputs[f"{name}_{key}"] = value
    return outputs


# The outputs are computed once per backend (scope="module", one fixture instance per param) and
# shared by all the parametrized test cases below, since computing them is the slow part.
# The "generic" backend (thecov.kernels) must reproduce the same reference as the original code.
@pytest.fixture(scope="module", params=["covapt", "generic"])
def outputs(request):
    return compute_all_outputs(request.param)


@pytest.fixture(scope="module")
def reference():
    return np.load(REFERENCE_FILE)


# One test case per backend, block and output (12 in total), so that a failure says which one changed.
@pytest.mark.parametrize("name", ["gggg", "pppp"])
@pytest.mark.parametrize("key", ["WinKernel", "cov", "shotnoise"])
def test_gaussian_regression(outputs, reference, name, key):
    actual = outputs[f"{name}_{key}"]
    expected = reference[f"{name}_{key}"]

    assert actual.shape == expected.shape
    # rtol allows only floating-point round-off (e.g. from reordering the arithmetic in a refactor).
    # Some kernel entries are exactly zero, so the absolute tolerance is set relative to the largest entry
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max()
    )


# Running this file as a script regenerates the reference file from the current code.
if __name__ == "__main__":
    os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
    np.savez(REFERENCE_FILE, **compute_all_outputs("covapt"))
    print(f"Saved reference outputs to {REFERENCE_FILE}.")
