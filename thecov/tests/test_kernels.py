"""Fast tests of thecov/kernels.py (a few seconds in total).

The agreement with the original window kernel code for C_gggg and C_pppp is tested in
test_regression_gaussian.py (kernel_backend="generic"). These tests cover what that one cannot:
the odd multipoles needed by the galaxy-momentum cross covariances, and the choice of blocks.
"""

import numpy as np
import pytest
from numpy.polynomial import legendre as npleg

import thecov.geometry
import thecov.kernels as K


# ---------------------------------------------------------------------------------------------
# Blocks and kernel types
# ---------------------------------------------------------------------------------------------


def test_blocks_use_every_kernel_type_once():
    """Every kernel type in ALL_KERNELS belongs to exactly one block, and every name exists."""
    names = [name for block in K.BLOCKS.values() for name in block]
    assert sorted(names) == sorted(spec.name for spec in K.ALL_KERNELS)


@pytest.mark.parametrize("block", list(K.BLOCKS))
def test_check_block_accepts_independent_blocks(block):
    K.check_block(block)


@pytest.mark.parametrize("block, match", [("ggpp", "transpose of ppgg"), ("gpgp", "Unknown block")])
def test_check_block_rejects_other_blocks(block, match):
    with pytest.raises(ValueError, match=match):
        K.check_block(block)


def test_cli_rejects_kernels_outside_the_block():
    """--kernels must be a subset of the block's kernel types (checked before reading any file)."""
    args = ["compute", "--block", "gggg", "--kernels", "G_pg_gg", "--windows-dir", "unused",
            "--out-dir", "unused", "--kmin", "0", "--kmax", "0.1", "--dk", "0.01"]
    with pytest.raises(ValueError, match="not part of block gggg"):
        K.main(args)


def test_geometry_rejects_unknown_kernel_backend():
    # __new__ avoids building a mesh: only the property setter is tested
    geometry = thecov.geometry.SurveyGeometry.__new__(thecov.geometry.SurveyGeometry)
    assert geometry.kernel_backend == "covapt"  # default, also for geometries saved before the option
    geometry.kernel_backend = "generic"
    with pytest.raises(ValueError, match="kernel_backend must be one of"):
        geometry.kernel_backend = "fast"


# ---------------------------------------------------------------------------------------------
# Windows times Legendre polynomials, including odd multipoles
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "k1_ells, k2_ells",
    [((1,), (3,)), ((3,), (1,)), ((3,), (3,)), ((1, 3), (2,)), ((3,), (1, 4)),
     ((4,), (2,)), ((2, 2), (4,)), ((1,), ()), ((), (3,)), ((3, 1), (1, 3))],
)
def test_projection_matches_direct_dft(k1_ells, k2_ells):
    """_Projector.side against a direct DFT of W(x) * prod of Legendre polynomials.

    With max_order at least the total degree, the Cartesian expansion is exact, so the two must
    agree to round-off. This checks the multinomial expansion, the Legendre products and the
    conventions (phase, which khat goes with which argument), for odd and even multipoles. The
    window is a random field on a small grid, so the test does not need pypower.
    """
    rng = np.random.default_rng(1)
    n = 6
    idx = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    pos = idx + rng.uniform(5, 9, size=3)  # observer outside the grid, so |x| > 0
    xhat = pos / np.linalg.norm(pos, axis=1)[:, None]
    W = rng.uniform(0.5, 1.5, size=len(idx))

    # Moments: FFTs of W xhat^gamma up to degree 8 (the highest total degree tested)
    max_order = 8
    moments = {"w": {}}
    for degree in range(max_order + 1):
        for exps in K.monomials(degree):
            weights = W * np.prod(xhat ** np.array(exps), axis=1)
            moments["w"][exps] = np.fft.fftn(weights.reshape(n, n, n))

    # Grid of q = k1 - k2 in the FFT order, a fixed k1, and khat2 at every grid point
    ik = np.fft.fftfreq(n, 1 / n)
    q = np.stack(np.meshgrid(ik, ik, ik, indexing="ij"), axis=0).reshape(3, -1)
    k1 = np.array([2.0, -1.0, 3.0])
    k1hat = k1 / np.linalg.norm(k1)
    k2 = k1[:, None] - q
    k2hat = k2 / np.linalg.norm(k2, axis=0)

    proj = K._Projector(moments, k1hat, k2hat, np.arange(q.shape[1]), max_order=max_order)
    got = proj.side("w", k1_ells, k2_ells)

    # Direct sum over the grid points x of W(x) f(xhat) exp(-i q.x), for every q
    def legendre(ell, t):
        return npleg.legval(t, np.eye(ell + 1)[ell])

    f = np.ones((q.shape[1], len(idx)))
    for ell in k1_ells:
        f = f * legendre(ell, xhat @ k1hat)[None, :]
    for ell in k2_ells:
        f = f * legendre(ell, k2hat.T @ xhat.T)
    phase = np.exp(-2j * np.pi * (q.T @ idx.T) / n)
    direct = np.sum(phase * f * W[None, :], axis=1)

    np.testing.assert_allclose(got, direct, rtol=0, atol=1e-12 * np.abs(direct).max())


def test_phase_sign():
    """(-1)^(sum of the orders on the flipped khat), e.g. for M_pg_g12 with l1 = 1, l2 = 0, l' = 1."""
    spec = K.get_kernel("M_pg_g12")
    sign = K.phase_sign(spec, "-(k1+k2)")
    # which = 0, "(2-1)": only l2 = 0 is on khat2 -> +1; which = 1, "(2-2)": l' + l2 = 1 -> -1
    np.testing.assert_array_equal(sign[0, 0, 0, 0], 1)
    np.testing.assert_array_equal(sign[0, 0, 0, 1], -1)
    assert np.all(K.phase_sign(spec, "-(k1-k2)") == 1)
