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


# ---------------------------------------------------------------------------------------------
# k bins when kmin is not a multiple of dk
# ---------------------------------------------------------------------------------------------


class _OnesProjector:
    """Stand-in for _Projector whose window FFTs are all 1, so that every kernel entry of a k2
    bin is just the number of k2 grid modes in that bin."""

    def __init__(self, moments, k1hat, k2hat, sel, max_order):
        self.n = len(sel)

    def side(self, window, k1_ells, k2_ells):
        return np.ones(self.n, dtype=complex)


@pytest.mark.parametrize(
    "kmin, kmax, dk",
    # 0.015: not a multiple of dk. 0.03: a multiple, but 0.03 // 0.01 = 2.0 in floating point.
    [(0.0, 0.04, 0.01), (0.015, 0.055, 0.01), (0.03, 0.07, 0.01)],
)
def test_k2_bins_match_k1_bins(monkeypatch, kmin, kmax, dk):
    """The k2 modes are binned in the same [kmin + j dk, kmin + (j+1) dk) bins as the k1 modes
    and Nmodes of sample_kmodes.

    With all window FFTs equal to 1, the kernel of a k2 bin is (number of k2 grid modes in the
    bin) / Nmodes. The grid of k2 = k1 - q covers every lattice mode below kmax (2 kmax / kfun <
    nmesh / 2 - 1), and Nmodes are the exact lattice counts, so the kernel must be exactly 1 for
    every k2 bin of the covariance. A shifted k2 binning gives a different count.
    """
    monkeypatch.setattr(K, "_Projector", _OnesProjector)
    boxsize, nmesh = 1000.0, 48
    sampled = K.math_mc.sample_kmodes(
        kmin=kmin, kmax=kmax, dk=dk, boxsize=boxsize, max_modes=10, rng=0, exact_nmodes=True,
        # k_shell_approx = 0: every bin is enumerated as a shell. With the default 0.1,
        # kmax < 0.1 hits a known bug of sample_kmodes, which then returns an extra bin.
        k_shell_approx=0.0,
    )
    kbins = len(sampled[0])
    assert kbins == round((kmax - kmin) / dk)
    assert 2 * kmax / (2 * np.pi / boxsize) < nmesh / 2 - 1

    out = K.compute_kernel(
        "N_g12_g12", kmin, kmax, dk, moments={"g12": {}}, boxsize=boxsize, nmesh=nmesh,
        sampled=sampled, delta_k_max=kbins - 1,
    )
    kernel = out["kernel"][..., 0, 0, 0]  # entry (l1, l2, placement) = (0, 0, (1,1))
    for i in range(kbins):
        for d in range(2 * kbins - 1):
            j = i + d - (kbins - 1)  # k2 bin
            expected = 1.0 if 0 <= j < kbins else 0.0
            np.testing.assert_allclose(kernel[i, d], expected, rtol=1e-12, err_msg=f"k1 bin {i}, k2 bin {j}")


def test_backends_agree_when_kmin_is_not_a_multiple_of_dk():
    """The original window kernel code (kernel_backend="covapt") bins k2 in the same way."""
    from thecov.tests.test_regression_gaussian import ALPHA, make_randoms

    WinKernel = {}
    for backend in ("covapt", "generic"):
        geometry = thecov.geometry.SurveyGeometry(
            make_randoms(velocity=False), nmesh=16, alpha=ALPHA, kmodes_sampled=10,
            nthreads=1, kernel_backend=backend,
        )
        # kmin is half a bin. kmax > 0.1 because of the known kmax < 0.1 bug of sample_kmodes.
        geometry.set_kbins(0.006, 0.126, 0.012)
        geometry.compute_window_kernels()
        WinKernel[backend] = geometry.WinKernel
    scale = np.abs(WinKernel["covapt"]).max()
    np.testing.assert_allclose(WinKernel["generic"], WinKernel["covapt"], rtol=0, atol=1e-12 * scale)


# ---------------------------------------------------------------------------------------------
# Wavevectors of the FFT grid, for odd and even nmesh
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nmesh, expected",
    [(5, [0, 1, 2, -2, -1]), (6, [0, 1, 2, -3, -2, -1]), (49, None), (64, None)],
)
def test_ikgrid_matches_fft_frequencies(nmesh, expected):
    """Both backends use the integer frequencies of np.fft.fftfreq, including for odd nmesh."""
    iik = np.arange(nmesh)
    iik = np.where(iik < (nmesh + 1) // 2, iik, iik - nmesh)  # integer FFT frequencies
    if expected is not None:
        np.testing.assert_array_equal(iik, expected)

    # SurveyGeometry.ikgrid (__new__: only the property is tested, no mesh is built)
    geometry = thecov.geometry.SurveyGeometry.__new__(thecov.geometry.SurveyGeometry)
    geometry.nmesh = nmesh
    for axis in geometry.ikgrid:
        np.testing.assert_array_equal(axis, iik)

    # kernels._ikgrid: flattened in the C order of fft.reshape(-1), axis 0 = x
    grid = np.stack(np.meshgrid(iik, iik, iik, indexing="ij")).reshape(3, -1)
    np.testing.assert_array_equal(K._ikgrid(nmesh), grid)


# ---------------------------------------------------------------------------------------------
# Empty k bins
# ---------------------------------------------------------------------------------------------


def test_empty_k_bins_raise():
    """With dk < kfun, some low-k bins contain no lattice mode: compute_kernel raises an error
    naming them before computing anything (otherwise the normalisation gives 0 / 0 = NaN)."""
    # boxsize = 1000: kfun = 0.00628, so [0.002, 0.004) and [0.004, 0.006) have no mode.
    # kmax > 0.1 because of the known kmax < 0.1 bug of sample_kmodes.
    with pytest.raises(ValueError, match=r"^2 k bin\(s\).*\[0\.002, 0\.004\), \[0\.004, 0\.006\)"):
        K.compute_kernel(
            "N_g12_g12", 0.0, 0.12, 0.002, moments={"g12": {}}, boxsize=1000.0, nmesh=16,
            kmodes_sampled=10,
        )


def test_check_nonempty_bins_accepts_full_bins():
    sampled = K.math_mc.sample_kmodes(0.0, 0.12, 0.012, 1000.0, max_modes=10, rng=0)
    K.check_nonempty_bins(*sampled, 0.0, 0.012, 1000.0)


def test_empty_k_bins_raise_in_covapt_backend():
    """The original window kernel code checks for empty bins too."""
    from thecov.tests.test_regression_gaussian import ALPHA, make_randoms

    geometry = thecov.geometry.SurveyGeometry(
        make_randoms(velocity=False), nmesh=16, alpha=ALPHA, kmodes_sampled=10, nthreads=1
    )
    # kfun ~ 0.0039 for this box, so with dk = 0.001 the bins between 0.001 and kfun are empty
    assert 2 * np.pi / geometry.boxsize > 0.003
    geometry.set_kbins(0.0, 0.12, 0.001)
    with pytest.raises(ValueError, match="contain no Fourier modes"):
        geometry.compute_window_kernels()


# ---------------------------------------------------------------------------------------------
# Window ids: prepare, compute and merge refuse to mix different windows
# ---------------------------------------------------------------------------------------------

# Tiny setup: a few hundred points on an 8^3 mesh, one kernel type (N_g12_g12, window "g12")
_BOX = dict(nmesh=8, boxsize=1000.0, boxcenter=[0.0, 0.0, 0.0], max_order=2)
# kmax > 0.1 because of the known kmax < 0.1 bug of sample_kmodes
_KBINS = ["--kmin", "0.0", "--kmax", "0.12", "--dk", "0.012"]


def _window(seed):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(100, 300, size=(300, 3))
    return {"g12": (positions, rng.uniform(0.5, 1.5, size=300))}


def _compute(windows_dir, out_dir, *extra):
    K.main(["compute", "--block", "gggg", "--kernels", "N_g12_g12", "--windows-dir",
            str(windows_dir), "--out-dir", str(out_dir), *_KBINS, "--kmodes-sampled", "2",
            "--max-order", "2", *extra])


def test_prepare_refuses_existing_windows(tmp_path):
    meta = K.prepare_windows(_window(0), tmp_path, **_BOX)
    assert set(meta["windows"]["g12"]) == {"I", "n_objects", "sha256"}
    assert meta["mesh_kwargs"] == {"interlacing": 3, "resampler": "tsc", "compensate": True}

    # Same windows again: nothing to do
    with pytest.raises(ValueError, match="already prepared"):
        K.prepare_windows(_window(0), tmp_path, **_BOX)

    # Different windows: refused, naming the differing fields, unless overwrite=True
    with pytest.raises(ValueError, match=r"windows\.g12\.I.*windows\.g12\.sha256"):
        K.prepare_windows(_window(1), tmp_path, **_BOX)
    new = K.prepare_windows(_window(1), tmp_path, overwrite=True, **_BOX)
    assert new["id"] != meta["id"]
    assert K.read_windows_meta(tmp_path)["id"] == new["id"]


def test_windows_id_ignores_float32_and_information_fields(tmp_path):
    """The same catalogue in float32 and float64 has the same id (when the float32 values are
    exactly representable), and created / thecov_commit are not part of it."""
    positions, weights = _window(0)["g12"]
    positions = positions.astype("f4").astype("f8")
    a = K.prepare_windows({"g12": (positions, weights)}, tmp_path / "a", **_BOX)
    b = K.prepare_windows({"g12": (positions.astype("f4"), weights)}, tmp_path / "b", **_BOX)
    assert a["id"] == b["id"]
    assert K.windows_id({**a, "created": "another time"}) == a["id"]


def test_compute_skips_identical_chunks_and_merge_rejects_different_windows(tmp_path, capsys):
    K.prepare_windows(_window(0), tmp_path / "wA", **_BOX)
    K.prepare_windows(_window(1), tmp_path / "wB", **_BOX)
    out = tmp_path / "kernels"

    # Chunk 0 from windows A; running the same task again skips it
    _compute(tmp_path / "wA", out, "--n-k1-chunks", "2", "--task-id", "0")
    (path,) = out.glob("N_g12_g12_k1bins_*.npz")
    meta = K.load_kernel(path)["meta"]
    assert meta["windows_id"] == K.read_windows_meta(tmp_path / "wA")["id"]
    mtime = path.stat().st_mtime_ns
    _compute(tmp_path / "wA", out, "--n-k1-chunks", "2", "--task-id", "0")
    assert "skipping" in capsys.readouterr().out
    assert path.stat().st_mtime_ns == mtime

    # Same chunk from windows B: refused, the file is not overwritten
    with pytest.raises(ValueError, match="differing fields: .*windows_id"):
        _compute(tmp_path / "wB", out, "--n-k1-chunks", "2", "--task-id", "0")

    # Chunk 1 from windows B: merge refuses to combine it with chunk 0 from windows A
    _compute(tmp_path / "wB", out, "--n-k1-chunks", "2", "--task-id", "1")
    with pytest.raises(ValueError, match="different settings or windows.*windows_id"):
        K.merge_kernel_files(str(out), "N_g12_g12")

    # Both chunks from the same windows merge fine
    out2 = tmp_path / "kernels2"
    _compute(tmp_path / "wA", out2, "--n-k1-chunks", "2")
    K.merge_kernel_files(str(out2), "N_g12_g12")
