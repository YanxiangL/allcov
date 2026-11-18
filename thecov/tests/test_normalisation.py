"""Tests of the power spectrum normalisation pk_renorm in thecov/covariance.py.

Convention: the measured power spectrum and the shotnoise property are in the normalisation of
the measured spectrum, P_meas = pk_renorm * P_thecov, where P_thecov is normalised by I22 of the
randoms (the normalisation the window kernels assume). get_pk returns P_meas / pk_renorm, and
_build_covariance_survey multiplies the covariance by (pk_renorm / I22)^2 once.

The regression test uses pk_renorm = 1, so it cannot catch a misplaced pk_renorm factor. These
tests use pk_renorm != 1.
"""

import numpy as np
import pytest

import thecov.covariance
import thecov.geometry
from thecov.tests.test_regression_gaussian import ALPHA, DK, KMAX, KMIN, make_randoms

PK_RENORM = 1.05


@pytest.fixture(scope="module")
def survey_geometry():
    """Small toy survey geometry (galaxies). Window kernels are computed on first use."""
    return thecov.geometry.SurveyGeometry(
        make_randoms(velocity=False), nmesh=16, alpha=ALPHA, kmodes_sampled=20, nthreads=1
    )


def test_get_pk_adds_shotnoise_in_the_same_normalisation():
    """get_pk(0, remove_shotnoise=False) = (P_meas + shotnoise) / pk_renorm."""
    geometry = thecov.geometry.BoxGeometry(volume=1e9, nbar=1e-4)
    cov = thecov.covariance.GaussianCovariance(geometry)
    cov.set_kbins(0.0, 0.1, 0.01)
    cov.pk_renorm = PK_RENORM

    P0 = np.linspace(2e4, 1e4, cov.kbins)
    cov.set_galaxy_pk_multipole(P0, 0, has_shotnoise=False)

    # shotnoise property = pk_renorm / nbar (measured normalisation)
    np.testing.assert_allclose(cov.shotnoise, PK_RENORM * 1e4)
    np.testing.assert_allclose(
        cov.get_pk(0, remove_shotnoise=False), (P0 + PK_RENORM * 1e4) / PK_RENORM
    )


def test_shotnoise_rescaling_func_matches_covariance(survey_geometry):
    """At the alpha used by compute_covariance, the covariance of the shot-noise rescaling
    function equals self.cov, so its likelihood derivative vanishes (it used to multiply the
    mixed and shot-noise terms by extra factors of pk_renorm)."""
    cov = thecov.covariance.GaussianCovariance(survey_geometry)
    cov.set_kbins(KMIN, KMAX, DK)
    k = cov.kmid
    P0 = 1e4 * 0.02 / (k + 0.01)
    cov.set_galaxy_pk_multipole(P0, 0)
    cov.set_galaxy_pk_multipole(0.5 * P0, 2)
    cov.set_galaxy_pk_multipole(0.1 * P0, 4)
    cov.pk_renorm = PK_RENORM
    cov.compute_covariance()

    # Only reference.cov is used: a copy of the covariance, so (reference - covariance) = 0
    # at alpha = cov.alpha
    reference = type("Reference", (), {"cov": cov.cov.copy()})()

    dlikelihood = cov._get_shotnoise_rescaling_func(reference)
    assert abs(dlikelihood(cov.alpha)) < 1e-8 * abs(dlikelihood(2 * cov.alpha))


class _MockPypower:
    """Minimal stand-in for pypower's PowerSpectrumMultipoles, as used by load_pypower."""

    def __init__(self, P, wnorm, shotnoise, alpha):
        self.kedges = np.linspace(KMIN, KMAX, len(P[0]) + 1)
        self._P = P
        self.wnorm = wnorm
        self.shotnoise = shotnoise
        self.attrs = {"sum_data_weights1": alpha * 1000.0, "sum_randoms_weights1": 1000.0}

    def __getitem__(self, item):
        sliced = _MockPypower([p[item] for p in self._P], self.wnorm, self.shotnoise, 1.0)
        sliced.attrs = self.attrs
        return sliced

    def get_power(self, remove_shotnoise=True, complex=False):
        return [p.copy() for p in self._P]


def _mock(geometry, renorm):
    """Mock pypower spectrum whose wnorm gives pk_renorm = renorm."""
    P0 = np.linspace(2e4, 1e4, round((KMAX - KMIN) / DK))
    return _MockPypower([P0, 0.5 * P0, 0.1 * P0], geometry.I("22") / renorm, 3e3, ALPHA)


@pytest.mark.parametrize("set_shotnoise", [False, True])
def test_load_pypower_normalisation(survey_geometry, set_shotnoise):
    """On a fresh covariance object (alpha not set yet): pk_renorm comes from wnorm and, with
    set_shotnoise=True, is then rescaled so that the shot noise matches pypower's."""
    pypower = _mock(survey_geometry, PK_RENORM)
    cov = thecov.covariance.GaussianCovariance(survey_geometry)
    cov.load_pypower(pypower, set_shotnoise=set_shotnoise)

    np.testing.assert_allclose(cov.alpha, ALPHA)
    if set_shotnoise:
        np.testing.assert_allclose(cov.shotnoise, pypower.shotnoise)
    else:
        np.testing.assert_allclose(cov.pk_renorm, PK_RENORM)


def test_load_pypower_removes_shotnoise_with_final_normalisation(survey_geometry):
    """With remove_shotnoise=False, thecov subtracts its own shot noise estimate, computed with
    the alpha and pk_renorm of this pypower file."""
    pypower = _mock(survey_geometry, PK_RENORM)
    cov = thecov.covariance.GaussianCovariance(survey_geometry)
    cov.load_pypower(pypower, remove_shotnoise=False)

    expected = PK_RENORM * (1 + ALPHA) * survey_geometry.I("12") / survey_geometry.I("22")
    np.testing.assert_allclose(pypower._P[0] - cov._pk[0], expected)
