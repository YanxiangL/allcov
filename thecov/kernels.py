"""Window kernels for the joint Gaussian covariance of the galaxy (gg), galaxy-momentum (pg) and
momentum (pp) power spectrum multipoles.

STANDALONE MODULE: nothing in thecov imports it yet. SurveyGeometry.compute_window_kernels still
computes the gggg / pppp kernels with its own (CovaPT) code. This module is the generic version
that computes every kernel needed by the joint covariance, and saves them to disk, so that the
covariance can later be assembled from the kernels and the power spectra without recomputing
them. Equation numbers refer to the notes "Analytical covariance matrix of peculiar velocity
power spectrum, peculiar velocity-galaxy number density cross power spectrum and galaxy power
spectrum".

Overview
--------
The calculation is split in three stages, so that the expensive part can run as a job array:

1. prepare  (prepare_windows): paint the windows on one common mesh and save the FFTs of
   W(x) * xhat^gamma for all Cartesian monomials xhat^gamma up to degree max_order. These are the
   "moments" of the window (W22, W22xx, W22xxyz, ... in SurveyGeometry).
2. kernels  (compute_kernel / the "compute" command): for one kernel type (see ALL_KERNELS) and a
   range of k1 bins, compute the kernels by Monte Carlo over sampled k1 modes and save them to
   one file per task.
3. merge    (merge_kernel_files / the "merge" command): concatenate the files of each kernel type
   over the k1 bins. Assembling the covariance (multiplying by the power spectra and the
   normalisations) is NOT done here.

The kernels do not depend on cosmology, the power spectra or the normalisations I_22: those are
all applied at assembly. So the same kernels can be reused for any input power spectrum.

The primitive kernel
--------------------
Every kernel in the notes (Gaussian eqs. 1.11-1.16, shot-noise eqs. 2.29-2.31, 2.35-2.37,
2.42-2.46, 2.54-2.55, 2.63-2.64) is a sum of terms of the form

    K(k1, k2) = < int_{x1, x2} exp[-i (k1 - k2).(x1 - x2)]
                  W_A(x1) f_1(xhat1; khat1, khat2)  W_B(x2) f_2(xhat2; khat1, khat2) >

where < > averages k1 over the k1 bin and k2 over the k2 bin, W_A is the window at x1, W_B the
window at x2, and f_1, f_2 are products of Legendre polynomials L_l(xhat . khat1) and
L_l(xhat . khat2). With F(q) = int_x W(x) f(xhat) exp(-i q.x) and q = k1 - k2 this is

    K = < Re[ F_A(q) conj(F_B(q)) ] >.

The real part is taken because the shell average of every kernel needed here is real: the total
Legendre order of every term is even (odd l of pg always come in pairs), so the imaginary part is
Monte Carlo noise only. This matches the CovaPT code, which also keeps the real part.

Other phases are not computed separately. Let K[phi] be the kernel computed with
exp[phi.(x1 - x2)] in place of the stored exp[-i(k1 - k2).(x1 - x2)], with everything else
unchanged. The integrands differ, but the shell averages over k1 and k2 are symmetric under
k -> -k: the integral of f(k2) over a shell equals that of f(-k2). Evaluating the integrand at
-k2 (and/or -k1) turns the phase into the stored one, and each L_l with khat2 (khat1) in its
argument picks up (-1)^l, since L_l(-mu) = (-1)^l L_l(mu). The power spectrum multipoles depend
only on |k| and do not change. Hence
    K[-i(k1 + k2)] = (-1)^(sum of the l's on khat2) K[-i(k1 - k2)]
    K[+i(k1 + k2)] = (-1)^(sum of the l's on khat1) K[-i(k1 - k2)]
    K[+i(k1 - k2)] = (-1)^(sum of all the l's)       K[-i(k1 - k2)]
(see phase_sign). The sums include every Legendre polynomial with that khat in its argument, at
x1 or x2 (e.g. both l2 and l'2 for khat2 in a Gaussian kernel). This is exact for full shells
(the sum over the k2 grid modes), and holds up to Monte Carlo noise for the sampled k1 modes.

This sign is separate from any (-1)^l' in the coefficients of the notes, e.g. the one from
P_pd(-k2) = sum_l' (-1)^l' P_l'(k2) L_l' in eq. 2.19: assembly has to apply both, and they can
cancel (in the exp[-i(k1 + k2)] term of eq. 2.64 the total is (-1)^l' (-1)^(l' + l2) = +1).

How F(q) is computed, and the approximations
--------------------------------------------
For one sampled k1 mode, F is a linear combination of
    T_JL(q) = int_x W(x) L_J(xhat . khat1) L_L(xhat . khat2) exp(-i q.x),
where khat2 = (k1 - q)/|k1 - q| depends on the grid point q. The steps are:

* Products of Legendre polynomials with the same argument are reduced exactly into single
  Legendre polynomials (e.g. L2 L2 = 1/5 L0 + 2/7 L2 + 18/35 L4, numpy legmul).
* Multipoles above max_order are then dropped on each side ("per-argument truncation"). CovaPT
  does the same, e.g. W_k1L4_Sumk2L44 drops the L6 and L8 parts of L4 L4.
* T_JL with J + L <= max_order is computed exactly from the moments of degree <= J + L:
  (xhat.khat1)^m (xhat.khat2)^n is expanded with the multinomial theorem.
* T_JL with J + L > max_order would need moments of a higher degree. It is approximated with
  khat2 ~ khat1, which holds near the diagonal (|k1 - k2| << k), reducing L_J L_L into single
  multipoles of khat1 and again dropping those above max_order. This is the approximation CovaPT
  uses for W_k1L2_k2L4, W_k1L4_k2L4 and their W12 versions.

With max_order = 4 this reproduces the CovaPT window kernels of SurveyGeometry for gggg and pppp
term by term (see to_covapt_winkernel). Odd multipoles (pg) need odd moments (degree 1 and 3 for
max_order = 4); prepare_windows computes them. A larger max_order (e.g. 6) makes the expansion
exact to higher order, at the cost of more moments and a slower kernel calculation, and can be
used to check how much the approximation matters.

Normalisation
-------------
As in SurveyGeometry.compute_window_kernels, each kernel is averaged over the sampled k1 modes
(divided by their number) and summed over the k2 grid modes of each k2 bin divided by the number
of modes Nmodes in that bin (from math_mc.sample_kmodes). The factors (2l1+1)(2l2+1), the 1/2,
1/4, 1/8 of the notes, (1 + alpha), and the 1/(I_22 I_22) normalisation are NOT included.

Usage (job array)
-----------------
    # once, in Python (needs the catalogues; see prepare_windows and the weight helpers)
    prepare_windows({...}, "windows/", nmesh=..., boxsize=..., boxcenter=...)

    # list the kernel types of a block (BLOCKS) and the number of tasks of its job array
    python -m thecov.kernels list --block pgpg --n-k1-chunks 4

    # one job array per block, one task per (kernel type of the block, chunk of k1 bins);
    # SLURM_ARRAY_TASK_ID is used if --task-id is omitted
    python -m thecov.kernels compute --block pgpg --windows-dir windows/ --out-dir kernels/ \
        --kmin 0.0 --kmax 0.2 --dk 0.005 --n-k1-chunks 4 --task-id 0 --nthreads 32

C_gggg and C_pppp can also be computed with SurveyGeometry(kernel_backend="generic"), which
calls compute_kernel with the moments of the geometry. The original CovaPT code
(kernel_backend="covapt") only computes these two blocks.

    # after all tasks: one file per kernel type
    python -m thecov.kernels merge --out-dir kernels/
"""

from __future__ import annotations

import argparse
import glob
import itertools as itt
import json
import logging
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Mapping

import numpy as np
from numpy.polynomial import legendre as npleg

from . import math_mc

__all__ = [
    "ELLS_EVEN",
    "ELLS_ODD",
    "PLACEMENTS",
    "KernelSpec",
    "ALL_KERNELS",
    "get_kernel",
    "BLOCKS",
    "check_block",
    "w22_weights",
    "w12_weights",
    "common_box",
    "compute_window_moments",
    "prepare_windows",
    "load_window_moments",
    "compute_kernel",
    "save_kernel",
    "merge_kernel_files",
    "phase_sign",
    "to_covapt_winkernel",
]

logger = logging.getLogger("kernels")

# Multipoles of the auto spectra (gg, pp) and of the cross spectrum (pg), which only has odd
# multipoles (P^pg = i Q, eq. 1.1).
ELLS_EVEN = (0, 2, 4)
ELLS_ODD = (1, 3)

# Positions of the estimator Legendre polynomials (L_l1(xhat . khat1), L_l2(xhat . khat2)) in the
# shot-noise kernels: (1, 2) means L_l1 at x1 and L_l2 at x2. Index p of a placement in the
# kernel arrays is its position in this tuple. The brackets of the notes are sums over these:
#   gggg/pppp (2-1), (2-2) (eqs. 2.29, 2.30): all four
#   gggg/pppp (3)          (eq. 2.31):        (1, 1) and (1, 2) only, i.e. L_l1 at x1
#   pd-pd, pd-pp, pd-dd    (eqs. 2.42-2.46, 2.54, 2.55, 2.63, 2.64): all four
PLACEMENTS = ((1, 1), (1, 2), (2, 1), (2, 2))

# Tolerance below which a Legendre series coefficient is treated as zero
_ZERO = 1e-14


# =================================================================================================
# 1. Polynomial helpers
# =================================================================================================


@lru_cache(maxsize=None)
def monomials(degree: int) -> tuple[tuple[int, int, int], ...]:
    """Exponents (ex, ey, ez) of all Cartesian monomials x^ex y^ey z^ez of a given degree."""
    # ex runs from degree down to 0, then ey from what is left down to 0, and ez takes the rest,
    # so every (ex, ey, ez) with ex + ey + ez = degree appears once: (degree+1)(degree+2)/2 of them
    # (1, 3, 6, 10, 15 for degrees 0-4). The order is fixed, so file names and loops are reproducible.
    return tuple(
        (ex, ey, degree - ex - ey)
        for ex in range(degree, -1, -1)
        for ey in range(degree - ex, -1, -1)
    )


def monomial_label(exps: tuple[int, int, int]) -> str:
    """File label of a monomial, e.g. (2, 1, 1) -> 'xxyz', and (0, 0, 0) -> '1'.

    Same letters as the window labels of SurveyGeometry (W22xxyz), without the 'W22' prefix.
    """
    label = (
        "x" * exps[0] + "y" * exps[1] + "z" * exps[2]
    )  # A string times a number will duplicate the string.
    return label or "1"


def label_to_exps(label: str) -> tuple[int, int, int]:
    """Inverse of monomial_label."""
    if label == "1":
        return (0, 0, 0)
    return (label.count("x"), label.count("y"), label.count("z"))


@lru_cache(maxsize=None)
def multinomial(exps: tuple[int, ...]) -> int:
    """Multinomial coefficient n! / (e1! e2! ...), with n = sum(exps)."""
    # Integer arithmetic, so the coefficient is exact. It is the number of ways of ordering the
    # factors, e.g. (a.x)^2 contains x_x x_y twice (x_x x_y and x_y x_x): multinomial((1, 1)) = 2.
    out = math.factorial(sum(exps))
    for e in exps:
        out //= math.factorial(
            e
        )  # Calculating the coefficient of the multinomial term corresponding to this exponent.
    return out


@lru_cache(maxsize=None)
def legendre_power_coeffs(ell: int) -> np.ndarray:
    """Coefficients p_m of L_ell(t) = sum_m p_m t^m (m = 0..ell)."""
    # The Legendre series with a single 1 at position ell is L_ell itself; leg2poly converts it to
    # ordinary powers of t, e.g. ell = 2 gives [-1/2, 0, 3/2]. Only powers with the parity of ell
    # are non-zero, which is where the (-1)^ell parity of L_ell comes from.
    coeffs = np.zeros(ell + 1)
    coeffs[ell] = 1.0
    return npleg.leg2poly(coeffs)


@lru_cache(maxsize=None)
def legendre_product(ells: tuple[int, ...]) -> np.ndarray:
    """Legendre series c_L of prod_i L_{ells[i]}(t) = sum_L c_L L_L(t), all with the same argument t.

    Exact (up to round-off): e.g. legendre_product((2, 2)) = [1/5, 0, 2/7, 0, 18/35].
    The empty product is L_0 = 1.
    """
    # Start from L_0 = 1 and multiply by one L_ell at a time. legmul multiplies two Legendre series
    # and returns the result as a Legendre series, i.e. it applies the exact product rule
    # L_a L_b = sum_L (2L+1) (a b L; 0 0 0)^2 L_L.
    out = np.array([1.0])
    for ell in ells:
        single = np.zeros(ell + 1)
        single[ell] = 1.0
        out = npleg.legmul(out, single)
    # Coefficients that are zero by parity can come out as ~1e-17 from round-off; set them to
    # exactly 0 so the loops over coefficients skip them.
    out[np.abs(out) < _ZERO] = 0.0
    return out


# =================================================================================================
# 2. Kernel types
# =================================================================================================


@dataclass(frozen=True)
class KernelSpec:
    """One kernel type: which windows, which multipoles, and how the Legendre polynomials are placed.

    Attributes
    ----------
    name : str
        Name of the kernel type, also used in the output file names.
    kind : str
        "gaussian": cosmic-variance kernel, eq. 1.9 in the reduced form of eqs. 1.11-1.16.
            Array index (l1, l2, l'1, l'2, config). It multiplies P_l'1(k1) P_l'2(k2).
            config 0 (zeta = x2): x1 has L_l1(khat1) L_l'2(khat2);        x2 has L_l'1(khat1) L_l2(khat2).
            config 1 (zeta = x1): x1 has L_l1(khat1) L_l'2(khat2) L_l2(khat2); x2 has L_l'1(khat1).
            The bracket of eqs. 1.11-1.16 is config 0 + config 1.
        "mixed": P x shot-noise kernel, the (2-1) and (2-2) kernels of section 2.3.
            Array index (l1, l2, l', which, placement). It multiplies P_l'(k1) (which = 0, "(2-1)")
            or P_l'(k2) (which = 1, "(2-2)"). L_l'(xhat1 . khat_which) is at x1, with window1 (a
            W22); window2 (a W12) is at x2. placement: see PLACEMENTS.
        "shotnoise": pure shot-noise kernel, the (3) kernels of section 2.3.
            Array index (l1, l2, placement). Both windows are W12's.
    window1, window2 : str
        Windows at x1 (not conjugated) and x2 (conjugated). Names used by prepare_windows:
        "gg", "pp", "pg" for W22^gg, W22^pp, W22^pg, and "g12", "p12" for W12^g and W12^p <u^2>.
    ells1, ells2 : tuple of int
        Multipoles l1 (of the first spectrum, P^ab) and l2 (of the second, P^cd).
    ellsp1, ellsp2 : tuple of int
        gaussian: multipoles l'1 of the spectrum at k1, and l'2 of the spectrum at k2.
        mixed: ellsp1 are the multipoles l' of the spectrum in the (2-1)/(2-2) kernels; ellsp2 unused.
    used_in : str
        Which covariance blocks and equations of the notes use this kernel.
    """

    name: str
    kind: str
    window1: str
    window2: str
    ells1: tuple[int, ...]
    ells2: tuple[int, ...]
    ellsp1: tuple[int, ...] = ()
    ellsp2: tuple[int, ...] = ()
    used_in: str = ""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the kernel array for one (k1 bin, delta k), i.e. without the two k axes."""
        n1, n2 = len(self.ells1), len(self.ells2)
        if self.kind == "gaussian":
            return (n1, n2, len(self.ellsp1), len(self.ellsp2), 2)
        if self.kind == "mixed":
            return (n1, n2, len(self.ellsp1), 2, len(PLACEMENTS))
        if self.kind == "shotnoise":
            return (n1, n2, len(PLACEMENTS))
        raise ValueError(f"Unknown kernel kind {self.kind}.")

    @property
    def windows(self) -> tuple[str, ...]:
        """Distinct windows of this kernel type (one if window1 == window2), in order."""
        # dict.fromkeys removes duplicates while keeping the order (a set would not keep it)
        return tuple(dict.fromkeys([self.window1, self.window2]))

    def terms(self) -> Iterator[tuple[tuple[int, ...], tuple, tuple]]:
        """Yields (index, side1, side2) for every entry of the kernel array.

        side = (Legendre orders on khat1, Legendre orders on khat2) at that point, so that
        side1 at x1 multiplies window1 and side2 at x2 multiplies window2.
        """
        # Each (i, l) pair from enumerate is (position of the multipole in the array, multipole).
        # The index tuple uses the positions, and the sides use the multipoles themselves.
        if self.kind == "gaussian":
            # Reduced form of eqs. 1.11-1.16: x1 carries L_l1(xhat1.khat1) and the Legendre
            # polynomial of the spectrum at k2, L_l'2(xhat1.khat2); x2 carries that of the spectrum
            # at k1, L_l'1(xhat2.khat1). The two terms of the bracket differ only in where
            # L_l2(xhat.khat2) is.
            for (
                (i1, l1),
                (i2, l2),
                (j1, lp1),
                (j2, lp2),
            ) in itt.product(  # iterate over all combinations of indices and Legendre orders
                enumerate(self.ells1),
                enumerate(self.ells2),
                enumerate(self.ellsp1),
                enumerate(self.ellsp2),
            ):
                # config 0 (zeta = x2): L_l2 at x2
                yield (i1, i2, j1, j2, 0), ((l1,), (lp2,)), ((lp1,), (l2,))
                # config 1 (zeta = x1): L_l2 at x1
                yield (i1, i2, j1, j2, 1), ((l1,), (lp2, l2)), ((lp1,), ())
        elif self.kind == "mixed":
            # (2-1) / (2-2) kernels, e.g. eqs. 2.29, 2.30: the W22 window at x1 carries
            # L_l'(xhat1.khat1) (which = 0) or L_l'(xhat1.khat2) (which = 1). L_l1 (always with
            # khat1) and L_l2 (always with khat2) are at x1 or x2, as given by the placement.
            for (i1, l1), (i2, l2), (j, lp), which, (p, (pos1, pos2)) in itt.product(
                enumerate(self.ells1),
                enumerate(self.ells2),
                enumerate(self.ellsp1),
                (0, 1),
                enumerate(PLACEMENTS),
            ):
                # Tuples of the Legendre orders at each point, split by the khat in the argument.
                # () means no Legendre polynomial (L_0 = 1).
                x1_k1 = ((lp,) if which == 0 else ()) + ((l1,) if pos1 == 1 else ())
                x1_k2 = ((lp,) if which == 1 else ()) + ((l2,) if pos2 == 1 else ())
                x2_k1 = (l1,) if pos1 == 2 else ()
                x2_k2 = (l2,) if pos2 == 2 else ()
                yield (i1, i2, j, which, p), (x1_k1, x1_k2), (x2_k1, x2_k2)
        elif self.kind == "shotnoise":
            # (3) kernels, e.g. eq. 2.31: two W12 windows and only the estimator Legendre
            # polynomials L_l1(khat1) and L_l2(khat2), each at x1 or x2.
            for (i1, l1), (i2, l2), (p, (pos1, pos2)) in itt.product(
                enumerate(self.ells1), enumerate(self.ells2), enumerate(PLACEMENTS)
            ):
                x1 = ((l1,) if pos1 == 1 else (), (l2,) if pos2 == 1 else ())
                x2 = ((l1,) if pos1 == 2 else (), (l2,) if pos2 == 2 else ())
                yield (i1, i2, p), x1, x2
        else:
            raise ValueError(f"Unknown kernel kind {self.kind}.")

    def to_dict(self) -> dict:
        """Plain dict (lists instead of tuples), to save in the JSON metadata of the output files
        and to pass to worker processes."""
        return {
            "name": self.name,
            "kind": self.kind,
            "window1": self.window1,
            "window2": self.window2,
            "ells1": list(self.ells1),
            "ells2": list(self.ells2),
            "ellsp1": list(self.ellsp1),
            "ellsp2": list(self.ellsp2),
            "used_in": self.used_in,
        }


E, O = ELLS_EVEN, ELLS_ODD

# Every kernel type needed by the 6 independent blocks of the joint covariance (gggg, pppp, ppgg,
# pgpg, pggg, pgpp; the other blocks are transposes). The window at x1 always carries the Legendre
# polynomial of the spectrum at k2 (L_l'2(xhat1 . khat2)) and the window at x2 that of the
# spectrum at k1 (L_l'1(xhat2 . khat1)), as in eq. 1.9.
ALL_KERNELS: tuple[KernelSpec, ...] = (
    # ---- Gaussian (cosmic variance) kernels, eqs. 1.11-1.16 ----
    KernelSpec(
        "G_gg_gg",
        "gaussian",
        "gg",
        "gg",
        E,
        E,
        E,
        E,
        "gggg eq. 1.11: P^gg(k1) P^gg(k2)",
    ),
    KernelSpec(
        "G_pp_pp",
        "gaussian",
        "pp",
        "pp",
        E,
        E,
        E,
        E,
        "pppp eq. 1.12: P^pp(k1) P^pp(k2)",
    ),
    KernelSpec(
        "G_pg_pg_even",
        "gaussian",
        "pg",
        "pg",
        E,
        E,
        O,
        O,
        "ppgg eq. 1.13: P^gp(k1) P^pg(k2)",
    ),
    KernelSpec(
        "G_pg_pg_odd",
        "gaussian",
        "pg",
        "pg",
        O,
        O,
        O,
        O,
        "pgpg eq. 1.14, first term: P^pg(k1) P^pg(k2)",
    ),
    KernelSpec(
        "G_pp_gg",
        "gaussian",
        "pp",
        "gg",
        O,
        O,
        E,
        E,
        "pgpg eq. 1.14, second term: P^gg(k1) P^pp(k2)",
    ),
    KernelSpec(
        "G_pg_gg",
        "gaussian",
        "pg",
        "gg",
        O,
        E,
        E,
        O,
        "pggg eq. 1.15: P^gg(k1) P^pg(k2)",
    ),
    KernelSpec(
        "G_pp_pg",
        "gaussian",
        "pp",
        "pg",
        O,
        E,
        O,
        E,
        "pgpp eq. 1.16: P^pg(k1) P^pp(k2)",
    ),
    # ---- P x shot-noise kernels ((2-1) and (2-2)) ----
    KernelSpec(
        "M_gg_g12", "mixed", "gg", "g12", E, E, E, (), "gggg eqs. 2.29, 2.30: P^gg"
    ),
    KernelSpec(
        "M_pp_p12", "mixed", "pp", "p12", E, E, E, (), "pppp eqs. 2.35, 2.36: P^pp"
    ),
    KernelSpec(
        "M_gg_p12",
        "mixed",
        "gg",
        "p12",
        O,
        O,
        E,
        (),
        "pgpg eqs. 2.42, 2.43 (phase +(k1+k2)): P^dd",
    ),
    KernelSpec(
        "M_pp_g12",
        "mixed",
        "pp",
        "g12",
        O,
        O,
        E,
        (),
        "pgpg eqs. 2.44, 2.45 (phase -(k1+k2)): P^pp",
    ),
    KernelSpec(
        "M_pg_p12",
        "mixed",
        "pg",
        "p12",
        O,
        E,
        O,
        (),
        "pgpp eqs. 2.54, 2.55 (phases +(k1+k2) and +(k1-k2)): P^pd",
    ),
    KernelSpec(
        "M_pg_g12",
        "mixed",
        "pg",
        "g12",
        O,
        E,
        O,
        (),
        "pggg eqs. 2.63, 2.64 (phases -(k1+k2) and -(k1-k2)): P^pd",
    ),
    # ---- pure shot-noise kernels ((3)) ----
    KernelSpec("N_g12_g12", "shotnoise", "g12", "g12", E, E, (), (), "gggg eq. 2.31"),
    KernelSpec("N_p12_p12", "shotnoise", "p12", "p12", E, E, (), (), "pppp eq. 2.37"),
    KernelSpec(
        "N_p12_g12",
        "shotnoise",
        "p12",
        "g12",
        O,
        O,
        (),
        (),
        "pgpg eq. 2.46 (phase -(k1+k2))",
    ),
)
# ppgg has no shot-noise kernel (notes, end of section 2.3).

del E, O

# Kernel types needed by each independent block of the joint covariance. The other blocks are
# transposes: C^{cd,ab}(k2, k1) = C^{ab,cd}(k1, k2), see TRANSPOSED_BLOCKS.
BLOCKS: dict[str, tuple[str, ...]] = {
    "gggg": ("G_gg_gg", "M_gg_g12", "N_g12_g12"),
    "pppp": ("G_pp_pp", "M_pp_p12", "N_p12_p12"),
    "ppgg": ("G_pg_pg_even",),  # no shot-noise (end of section 2.3)
    "pgpg": ("G_pg_pg_odd", "G_pp_gg", "M_gg_p12", "M_pp_g12", "N_p12_g12"),
    "pggg": ("G_pg_gg", "M_pg_g12"),
    "pgpp": ("G_pp_pg", "M_pg_p12"),
}
TRANSPOSED_BLOCKS = {"ggpp": "ppgg", "ggpg": "pggg", "pppg": "pgpp"}


def check_block(block: str) -> None:
    """Raises a ValueError if block is not one of the independent blocks in BLOCKS.

    For a transposed block (e.g. ggpp) the message says which block to compute instead.
    """
    if block in BLOCKS:
        return
    if block in TRANSPOSED_BLOCKS:
        raise ValueError(
            f"Block {block} is the transpose of {TRANSPOSED_BLOCKS[block]}: compute "
            f"{TRANSPOSED_BLOCKS[block]} and transpose it at assembly."
        )
    raise ValueError(f"Unknown block {block!r}. Available: {list(BLOCKS)}.")


def get_kernel(name: str) -> KernelSpec:
    """Returns the kernel type with this name from ALL_KERNELS."""
    for spec in ALL_KERNELS:
        if spec.name == name:
            return spec
    raise KeyError(
        f"Unknown kernel {name}. Available: {[s.name for s in ALL_KERNELS]}."
    )


def phase_sign(spec: KernelSpec, phase: str) -> np.ndarray:
    """Sign s such that K[phase] = s * K[stored], for each entry of a kernel array.

    K[phase] is the kernel computed with exp[phase . (x1 - x2)] in place of the stored phase
    exp[-i(k1 - k2).(x1 - x2)], with everything else unchanged. The two integrands are different
    functions of k1, k2. The kernels are equal up to s only after the shell averages, which are
    symmetric under k -> -k: substituting k2 -> -k2 (and/or k1 -> -k1) in the average turns the
    phase into the stored one, and each L_l(xhat . khat) with the flipped khat gives (-1)^l
    (see the module docstring).

    Parameters
    ----------
    spec : KernelSpec
    phase : str
        One of "-(k1-k2)" (stored, sign +1), "-(k1+k2)", "+(k1+k2)", "+(k1-k2)", meaning
        exp[phase . (x1 - x2)] up to the factor i.

    Returns
    -------
    np.ndarray of +1/-1 with shape spec.shape, to multiply the kernel array with (broadcast over
    the k axes). It is (-1)^(sum of the l's on khat2) for "-(k1+k2)", (-1)^(sum on khat1) for
    "+(k1+k2)" and (-1)^(sum of all) for "+(k1-k2)". The sums are over every Legendre polynomial
    with that khat in its argument, at x1 or x2.

    This is only the sign from changing the integration variable. Coefficients of the notes
    that contain their own (-1)^l' (e.g. from P_pd(-k2), eq. 2.19) are not included and must be
    applied separately at assembly.
    """
    # Which of khat1 and khat2 are flipped (k -> -k) to turn the phase into the stored one
    flip = {
        "-(k1-k2)": (False, False),
        "-(k1+k2)": (False, True),
        "+(k1+k2)": (True, False),
        "+(k1-k2)": (True, True),
    }[phase]
    sign = np.ones(spec.shape)
    for index, side1, side2 in spec.terms():
        # side = (orders with khat1, orders with khat2). Add up the orders of each khat over both
        # points: flipping that khat multiplies the term by (-1)^(this sum).
        order_k1 = sum(side1[0]) + sum(side2[0])
        order_k2 = sum(side1[1]) + sum(side2[1])
        # A False flag multiplies the order by 0, so an unflipped khat contributes no sign
        sign[index] = (-1) ** (order_k1 * flip[0] + order_k2 * flip[1])
    return sign


# =================================================================================================
# 3. Windows (prepare stage)
# =================================================================================================
# The windows are sums over random points, with weights such that sum_i w_i delta(x - x_i) ~ W(x).
# A random catalogue of tracer a has density n_a(x) / alpha_a, so sum_i alpha_a f(x_i) ~ int n_a f.
# These follow SurveyGeometry.W_cat: W_ij = alpha * NZ^(i-1) * WEIGHT_FKP^j * WEIGHT.


def w22_weights(alpha_a, nz_b, fkp_a, fkp_b, weight_a=None) -> np.ndarray:
    """Weights of W22^ab(x) = n_a n_b w_a w_b, from the randoms of tracer a.

    Parameters
    ----------
    alpha_a : float
        Ratio of data to randoms of tracer a.
    nz_b : array
        Number density of tracer b at the positions of the randoms of a.
    fkp_a, fkp_b : array
        FKP weights of tracers a and b at the positions of the randoms of a.
    weight_a : array, optional
        Other weights (WEIGHT column) of the randoms of a.

    For a = b this is alpha * NZ * WEIGHT_FKP^2 * WEIGHT, the W22 of SurveyGeometry.

    For W22^pg, either random catalogue gives a Monte Carlo estimate of the same window, as long
    as the other tracer's number density and FKP weight are evaluated correctly at its positions,
    including zeros where the other tracer is absent. We use the momentum (PV) randoms (a = p,
    b = g), with the galaxy NZ and FKP weights at their positions. The window is zero outside the
    PV footprint and the PV randoms only sample inside it, so no points are wasted, and only n_g is
    needed at the PV positions. Using the galaxy randoms instead would need n_p, including the PV
    angular mask, at every galaxy random. Both estimates can also be combined by concatenating the
    two (positions, weights) sets with the weights halved.
    """
    weights = alpha_a * np.asarray(nz_b) * np.asarray(fkp_a) * np.asarray(fkp_b)
    if weight_a is not None:
        weights = weights * np.asarray(weight_a)
    return weights


def w12_weights(alpha, fkp, weight=None, var_u=None) -> np.ndarray:
    """Weights of the shot-noise window W12(x) = n w^2, times <u^2(x)> for the momentum field.

    alpha * WEIGHT_FKP^2 * WEIGHT (* VAR_U), the W12 of SurveyGeometry (with velocity=True if
    var_u is given, eq. 2.6).
    """
    weights = alpha * np.asarray(fkp) ** 2
    if weight is not None:
        weights = weights * np.asarray(weight)
    if var_u is not None:
        weights = weights * np.asarray(var_u)
    return weights


def common_box(*positions, boxpad: float = 2.0) -> tuple[float, np.ndarray]:
    """Cubic box (size, center) enclosing all position arrays, padded by boxpad.

    All windows must be painted on the same mesh (same boxsize, boxcenter and nmesh), because the
    kernels multiply their FFTs at the same wavevectors.
    """
    # Bounding box of all the catalogues together
    allpos = np.concatenate([np.asarray(p).reshape(-1, 3) for p in positions])
    pmin, pmax = allpos.min(axis=0), allpos.max(axis=0)
    # Cube whose side is the largest extent times boxpad. The padding (2 by default, as in
    # SurveyGeometry) avoids aliasing: the FFTs are periodic, so the window must not wrap around
    # onto itself when it is correlated with a shifted copy.
    boxsize = float(boxpad * np.max(pmax - pmin))
    return boxsize, (pmin + pmax) / 2.0


def compute_window_moments(
    positions,
    weights,
    nmesh: int,
    boxsize: float,
    boxcenter,
    max_order: int = 4,
    mesh_kwargs: dict | None = None,
) -> dict[tuple[int, int, int], np.ndarray]:
    """FFTs of W(x) * xhat^gamma for all monomials of degree 0..max_order.

    The mesh assignment is the same as SurveyGeometry.get_fft (pypower CatalogMesh, TSC with
    interlacing 3, compensated, multiplied by nmesh^3), so with the same box the FFTs are the
    same as those of SurveyGeometry. xhat = x / |x| with the observer at the origin, as in
    SurveyGeometry.compute_cartesian_ffts.

    Returns
    -------
    dict mapping the exponents (ex, ey, ez) to a complex array of shape (nmesh, nmesh, nmesh).
    """
    from pypower import CatalogMesh

    positions = np.asarray(positions)
    weights = np.asarray(weights, dtype="f8")
    # Unit vector of the line of sight to each point (observer at the origin), shape (N, 3)
    xhat = positions / np.sqrt(np.sum(positions**2, axis=-1))[:, None]
    # Same mass assignment as SurveyGeometry; mesh_kwargs can override it
    kwargs = {"interlacing": 3, "resampler": "tsc", **(mesh_kwargs or {})}

    moments = {}
    for degree in range(max_order + 1):
        for exps in monomials(degree):
            # Weight of each point times xhat_x^ex xhat_y^ey xhat_z^ez (xhat ** exps raises each
            # component to its exponent, and prod multiplies the three)
            w = weights * np.prod(xhat ** np.array(exps), axis=-1)
            # Paint the weighted points on the mesh. A new CatalogMesh with explicit boxsize,
            # boxcenter and nmesh for every moment guarantees that all windows share one mesh.
            mesh = CatalogMesh(
                data_positions=positions,
                data_weights=w,
                position_type="pos",
                nmesh=nmesh,
                boxsize=boxsize,
                boxcenter=boxcenter,
                dtype="c16",
                **kwargs,
            ).to_mesh(
                compensate=True
            )  # compensate: divide by the window of the TSC assignment
            fft = mesh.r2c()  # type: ignore
            # pypower normalises the FFT by 1/nmesh^3; undo it, so that the FFT is the plain sum
            # over the points, as in SurveyGeometry.get_fft
            fft *= nmesh**3
            # With dtype c16 the mesh is complex, so r2c returns the full (nmesh, nmesh, nmesh)
            # grid (not the half grid of a real FFT); .value is the numpy array
            moments[exps] = np.asarray(fft.value)
    return moments


def prepare_windows(
    windows: Mapping[str, tuple],
    out_dir: str,
    nmesh: int,
    boxsize: float,
    boxcenter,
    max_order: int = 4,
    mesh_kwargs: dict | None = None,
) -> dict:
    """Prepare stage: compute and save the moments of all windows on one common mesh.

    Parameters
    ----------
    windows : dict
        name -> (positions, weights). The kernel types in ALL_KERNELS use the names
        "gg", "pp", "pg" (W22, from w22_weights) and "g12", "p12" (W12, from w12_weights).
    out_dir : str
        Output directory. Writes <out_dir>/<name>/<label>.npy for each moment (label from
        monomial_label) and <out_dir>/windows.json with the mesh and the integrals I = sum of the
        weights of each window (I22^ab for W22^ab, I12 for W12), needed at assembly.
    nmesh, boxsize, boxcenter : mesh shared by all windows (see common_box).
    max_order : int
        Highest degree of the moments. Must be >= the max_order used in compute_kernel.

    Returns
    -------
    dict with the metadata written to windows.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "nmesh": int(nmesh),
        "boxsize": float(boxsize),
        "boxcenter": [float(c) for c in np.ravel(boxcenter)],
        "max_order": int(max_order),
        "windows": {},
    }
    for name, (positions, weights) in windows.items():
        t0 = time.time()
        moments = compute_window_moments(
            positions, weights, nmesh, boxsize, boxcenter, max_order, mesh_kwargs
        )
        # One .npy file per moment (e.g. windows/pg/xxyz.npy), so that a kernel task can load,
        # or memory-map, only the windows and degrees it needs
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        for exps, fft in moments.items():
            np.save(os.path.join(out_dir, name, monomial_label(exps) + ".npy"), fft)
        meta["windows"][name] = {
            # Sum of the weights = Monte Carlo estimate of int W(x) d^3x, e.g. I22^ab for W22^ab,
            # the same as SurveyGeometry.I("22"). Used for the normalisation at assembly.
            "I": float(np.sum(weights)),
            "n_objects": int(len(weights)),
        }
        logger.info(
            f"Window {name}: {len(moments)} moments in {time.time() - t0:.1f} s."
        )
    with open(os.path.join(out_dir, "windows.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_window_moments(
    windows_dir: str, name: str, max_order: int, mmap: bool = True
) -> dict[tuple[int, int, int], np.ndarray]:
    """Loads the moments of degree <= max_order of one window saved by prepare_windows.

    With mmap=True the arrays are memory-mapped, so parallel workers share the same pages
    through the operating system instead of each holding a copy.
    """
    moments = {}
    for degree in range(max_order + 1):
        for exps in monomials(degree):
            path = os.path.join(windows_dir, name, monomial_label(exps) + ".npy")
            # mmap_mode="r": the array is read from disk only when (and where) it is accessed
            moments[exps] = np.load(path, mmap_mode="r" if mmap else None)
    return moments


def moments_from_geometry(
    geometry, window: str
) -> dict[tuple[int, int, int], np.ndarray]:
    """Moments of a SurveyGeometry window ('22' or '12') in the format used here.

    Used to compare with the existing CovaPT kernels; SurveyGeometry only has the even moments
    (degrees 0, 2, 4).
    """
    geometry.W(f"W{window}xx")  # computes all the moments of this window if needed
    out = {}
    for degree in (0, 2, 4):
        for exps in monomials(degree):
            # SurveyGeometry stores them as "22", "22xx", "22xxyz", ... (no "1" for degree 0)
            label = window + ("" if degree == 0 else monomial_label(exps))
            out[exps] = geometry._W[label]
    return out


# =================================================================================================
# 4. Windows times Legendre polynomials, for one sampled k1 mode
# =================================================================================================


class _Projector:
    """Computes F(q) = int_x W(x) f(xhat) exp(-i q.x) for one sampled k1 mode.

    Only the grid points q that end in a k2 bin within delta_k_max of the k1 bin are kept
    (flat indices sel), so all arrays here are 1D over those points. Everything is cached, since
    the same pieces appear in many terms of a kernel.
    """

    def __init__(self, moments, k1hat, k2hat, sel, max_order):
        self.moments = moments  # window name -> {exps: FFT array (nmesh, nmesh, nmesh)}
        self.k1hat = np.asarray(k1hat, dtype="f8")  # (3,)
        self.k2hat = k2hat  # (3, nsel)
        self.sel = sel
        self.max_order = max_order
        # Caches, from the lowest level to the highest. They live only as long as this object,
        # i.e. for one k1 mode, since everything depends on khat1.
        self._mom = {}  # (window, exps) -> moment at the selected grid points
        self._k2mono = {}  # exps -> khat2^exps
        self._A = {}  # (window, m, n) -> A_mn
        self._T = {}  # (window, J, L) -> T_JL
        self._side = {}  # (window, k1 orders, k2 orders) -> F

    def moment(self, window, exps):
        """FFT of W(x) xhat^exps at the selected grid points."""
        key = (window, exps)
        if key not in self._mom:
            try:
                fft = self.moments[window][exps]
            except KeyError:
                raise KeyError(
                    f"Moment {monomial_label(exps)} of window {window} not available: compute "
                    f"the windows with max_order >= {sum(exps)}."
                ) from None
            # Flatten the 3D FFT and keep only the selected grid points. Fancy indexing makes a
            # copy, so with memory-mapped files only these points are read.
            self._mom[key] = np.asarray(fft).reshape(-1)[self.sel]
        return self._mom[key]

    def k2_monomial(self, exps):
        """khat2^exps at the selected grid points."""
        if exps not in self._k2mono:
            # khat2_x^ex khat2_y^ey khat2_z^ez, one value per selected grid point
            out = np.ones(self.k2hat.shape[1])
            for axis, e in enumerate(exps):
                if e:  # skip zero exponents (factor 1)
                    out = out * self.k2hat[axis] ** e
            self._k2mono[exps] = out
        return self._k2mono[exps]

    def A(self, window, m, n):
        """int_x W(x) (xhat.khat1)^m (xhat.khat2)^n exp(-i q.x), exact, from moments of degree m+n.

        Multinomial expansion: (xhat.khat)^m = sum_{|alpha|=m} m!/alpha! khat^alpha xhat^alpha.
        """
        key = (window, m, n)
        if key not in self._A:
            # (xhat.khat1)^m (xhat.khat2)^n
            #   = sum_beta n!/beta! khat2^beta [ sum_alpha m!/alpha! khat1^alpha xhat^(alpha+beta) ]
            # and int W xhat^(alpha+beta) exp(-iq.x) is the moment with exponents alpha + beta.
            # The inner sum over alpha has scalar coefficients (khat1 is fixed for this k1 mode);
            # the outer sum over beta has coefficients that vary over the grid (khat2 depends on q).
            out = np.zeros(self.k2hat.shape[1], dtype="c16")
            for beta in monomials(n):
                inner = None
                for alpha in monomials(m):
                    # Scalar: m!/alpha! * khat1_x^ax khat1_y^ay khat1_z^az
                    coeff = multinomial(alpha) * np.prod(self.k1hat ** np.array(alpha))
                    if coeff == 0.0:
                        # Happens when a component of khat1 is 0 (e.g. k1 along an axis, or the
                        # k1 = 0 mode where khat1 = 0); skipping saves an array operation
                        continue
                    exps = tuple(a + b for a, b in zip(alpha, beta))
                    term = coeff * self.moment(
                        window, exps
                    )  # new array, safe to add into
                    if inner is None:
                        inner = term
                    else:
                        inner += term
                if inner is None:  # every alpha term was zero
                    continue
                if n == 0:
                    # No khat2 dependence (beta = (0, 0, 0), khat2^beta = 1)
                    out += inner
                else:
                    out += (multinomial(beta) * self.k2_monomial(beta)) * inner
            self._A[key] = out
        return self._A[key]

    def T(self, window, J, L):
        """int_x W(x) L_J(xhat.khat1) L_L(xhat.khat2) exp(-i q.x).

        Exact if J + L <= max_order. Otherwise khat2 ~ khat1: L_J L_L is reduced to single
        multipoles K of khat1, keeping K <= max_order (the CovaPT approximation).
        """
        key = (window, J, L)
        if key not in self._T:
            out = np.zeros(self.k2hat.shape[1], dtype="c16")
            if J + L <= self.max_order:
                # Exact: write L_J(a) = sum_m pJ[m] a^m and L_L(b) = sum_n pL[n] b^n, with
                # a = xhat.khat1, b = xhat.khat2, so L_J(a) L_L(b) = sum_mn pJ[m] pL[n] a^m b^n
                # and each a^m b^n term is A(m, n). Needs moments up to degree J + L.
                # Example: L2(a) L2(b) = 9/4 a^2 b^2 - 3/4 a^2 - 3/4 b^2 + 1/4, CovaPT's
                # W_k1L2_k2L2 = 9/4 k1k2 - 3/4 xx - 1/2 W_k2L2.
                pJ, pL = legendre_power_coeffs(J), legendre_power_coeffs(L)
                for m in range(J + 1):
                    if pJ[m] == 0.0:  # powers with the wrong parity
                        continue
                    for n in range(L + 1):
                        if pL[n] == 0.0:
                            continue
                        out += (pJ[m] * pL[n]) * self.A(window, m, n)
            else:
                # Approximate: replace khat2 by khat1, so L_J(a) L_L(b) ~ L_J(a) L_L(a)
                # = sum_K c_K L_K(a) (exact product rule), and keep only K <= max_order, for
                # which T(K, 0) is exact. Example: L2 L4 = 2/7 L2 + 20/77 L4 (+ 5/11 L6, dropped),
                # CovaPT's W_k1L2_k2L4 = 2/7 W_k1L2 + 20/77 W_k1L4.
                coeffs = legendre_product((J, L))
                for K, c in enumerate(coeffs[: self.max_order + 1]):
                    if c != 0.0:
                        out += c * self.T(window, K, 0)
            self._T[key] = out
        return self._T[key]

    def side(self, window, k1_ells, k2_ells):
        """int_x W(x) prod L_l(xhat.khat1) [l in k1_ells] prod L_l(xhat.khat2) [l in k2_ells] exp(-i q.x)."""
        # Sorting makes e.g. (2, 4) and (4, 2) the same cache entry (the product is the same)
        key = (window, tuple(sorted(k1_ells)), tuple(sorted(k2_ells)))
        if key not in self._side:
            # Products with the same argument are reduced exactly, then multipoles above
            # max_order are dropped (per-argument truncation, as in CovaPT).
            # c1[J]: prod_l L_l(xhat.khat1) = sum_J c1[J] L_J(xhat.khat1); c2 likewise for khat2.
            c1 = legendre_product(key[1])[: self.max_order + 1]
            c2 = legendre_product(key[2])[: self.max_order + 1]
            # F = sum_JL c1[J] c2[L] T_JL
            out = np.zeros(self.k2hat.shape[1], dtype="c16")
            for J, a in enumerate(c1):
                if a == 0.0:
                    continue
                for L, b in enumerate(c2):
                    if b == 0.0:
                        continue
                    out += (a * b) * self.T(window, J, L)
            self._side[key] = out
        return self._side[key]


# =================================================================================================
# 5. Kernel calculation (kernels stage)
# =================================================================================================

# State of each worker process (set by _init_worker, or directly in the parent process).
_STATE: dict = {}


def _ikgrid(nmesh: int) -> np.ndarray:
    """Integer wavevectors of the FFT grid, flattened, shape (3, nmesh^3); as SurveyGeometry.ikgrid."""
    # FFT ordering: 0, 1, ..., nmesh/2 - 1, -nmesh/2, ..., -1 (in units of kfun), so that grid
    # index i of the FFT arrays corresponds to the wavevector iik[i]
    iik = np.arange(nmesh)
    iik[iik >= nmesh // 2] -= nmesh
    # indexing="ij" so that axis 0 is x, matching the FFT arrays; flattened in the same (C) order
    # as fft.reshape(-1)
    return np.array(
        [g.reshape(-1) for g in np.meshgrid(iik, iik, iik, indexing="ij")], dtype="f8"
    )


def _init_worker(windows_dir, spec_dict, params):
    """Pool initializer: memory-maps the moments of the windows of this kernel type."""
    # Runs once in each worker process. The spec is passed as a plain dict (lists from to_dict
    # converted back to tuples), and the moments are loaded here rather than sent from the
    # parent, which would copy every array to every worker.
    spec = KernelSpec(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in spec_dict.items()}  # type: ignore
    )
    _STATE["spec"] = spec
    _STATE["params"] = params
    _STATE["moments"] = {
        w: load_window_moments(windows_dir, w, params["max_order"])
        for w in spec.windows
    }
    _STATE["ikgrid"] = _ikgrid(params["nmesh"])


def _kernel_chunk(task):
    """Sum of the kernel over a chunk of k1 modes of one k1 bin (not yet normalised).

    Follows SurveyGeometry._compute_window_kernel_row: for each k1 mode, k2 = k1 - q runs over
    the whole FFT grid, and each k2 is assigned to bin floor(|k2| / dk).
    """
    ibin, k1_bin_index, kmodes = task
    spec, params, moments = _STATE["spec"], _STATE["params"], _STATE["moments"]
    ikgrid = _STATE["ikgrid"]
    kfun = (
        2 * np.pi / params["boxsize"]
    )  # Fundamental frequency of the FFT grid in units of 1/boxsize.
    dk, dkmax = params["dk"], params["delta_k_max"]
    ndelta = (
        2 * dkmax + 1
    )  # number of k2 bins kept: k1 bin - dkmax, ..., k1 bin + dkmax

    # acc[d, *index]: sum over the k1 modes of this chunk and over the k2 grid modes in the k2
    # bin (k1 bin + d - dkmax), of the kernel entry index. Normalised later in compute_kernel.
    acc = np.zeros((ndelta,) + spec.shape)
    for ik1x, ik1y, ik1z, ik1r in kmodes:
        # Sampled k1 mode, in units of kfun; khat1 = 0 for the k1 = 0 mode (as in SurveyGeometry)
        k1hat = np.zeros(3) if ik1r <= 1e-10 else np.array([ik1x, ik1y, ik1z]) / ik1r

        # The FFTs are evaluated at q = k1 - k2, and q runs over the grid, so k2 = k1 - q
        # covers a whole grid of k2 modes around k1. Shape (3, nmesh^3).
        k2 = np.array([ik1x, ik1y, ik1z])[:, None] - ikgrid
        k2r = np.sqrt(np.sum(k2**2, axis=0))
        # k2 bin floor(|k2| / dk), relative to the k1 bin, shifted by dkmax so that it is an
        # index 0..ndelta-1 for the k2 bins that are kept
        rel = (k2r * kfun / dk).astype(int) - int(k1_bin_index) + dkmax
        # Flat indices of the grid points whose k2 is in a kept bin; all others are dropped here,
        # so every array below has one value per selected point
        sel = np.flatnonzero((rel >= 0) & (rel < ndelta))
        idx = rel[sel]
        k2r_sel = k2r[sel]
        k2r_sel[k2r_sel <= 1e-10] = np.inf  # khat2 = 0 for k2 = 0, as in SurveyGeometry
        k2hat = k2[:, sel] / k2r_sel

        proj = _Projector(moments, k1hat, k2hat, sel, params["max_order"])
        for index, side1, side2 in spec.terms():
            # F at x1 (window1) and at x2 (window2) for this entry; most are reused from the cache
            F1 = proj.side(spec.window1, *side1)
            F2 = proj.side(spec.window2, *side2)
            # Re[F1 conj(F2)] = Re F1 Re F2 + Im F1 Im F2, without building the complex product
            prod = F1.real * F2.real + F1.imag * F2.imag
            # Sum prod over the grid points of each k2 bin in one pass: bincount adds weight
            # prod[i] to bin idx[i]. Same as looping over delta_k with masks, as SurveyGeometry
            # does, but much faster.
            acc[(slice(None),) + index] += np.bincount(
                idx, weights=prod, minlength=ndelta
            )
    return ibin, acc


def compute_kernel(
    spec: KernelSpec | str,
    kmin: float,
    kmax: float,
    dk: float,
    windows_dir: str | None = None,
    moments: Mapping | None = None,
    boxsize: float | None = None,
    nmesh: int | None = None,
    k1_bins: tuple[int, int] | None = None,
    kmodes_sampled: int = 5000,
    max_order: int = 4,
    delta_k_max: int | None = None,
    nthreads: int = 1,
    seed: int = 42,
    k_shell_approx: float = 0.1,
    sampled: tuple | None = None,
) -> dict:
    """Kernels stage: computes one kernel type for a range of k1 bins.

    Parameters
    ----------
    spec : KernelSpec or str
        Kernel type (or its name in ALL_KERNELS).
    kmin, kmax, dk : float
        k bins of the covariance, in h/Mpc.
    windows_dir : str
        Directory written by prepare_windows. boxsize and nmesh are read from it.
    moments : dict, optional
        Instead of windows_dir: window name -> moments dict, already in memory (e.g. from
        moments_from_geometry). Then boxsize and nmesh must be given.
    k1_bins : (int, int), optional
        Range [start, stop) of k1 bins to compute. Default: all bins.
    kmodes_sampled : int
        Maximum number of k1 modes sampled per bin (Monte Carlo over k1).
    max_order : int
        Highest degree of the Cartesian expansion (see the module docstring). 4 reproduces CovaPT.
    delta_k_max : int, optional
        Number of k2 bins kept on each side of the k1 bin. Default nmesh // 2 - 1, as in
        SurveyGeometry.
    nthreads : int
        Number of worker processes. With moments in memory, more than 1 needs the 'fork' start method.
    seed : int
        Seed of math_mc.sample_kmodes. The same seed (42, as in SurveyGeometry) in every task
        gives the same k1 modes, so the chunks of a job array are consistent.
    sampled : (kmodes, Nmodes), optional
        Output of math_mc.sample_kmodes, if already computed for these k bins, to avoid sampling
        again in every call (e.g. SurveyGeometry calls this once per k1 bin and kernel type).
        It must come from the same kmin, kmax, dk, boxsize, kmodes_sampled and seed.

    Returns
    -------
    dict with
        kernel : array (n_k1_bins, 2 * delta_k_max + 1, *spec.shape); the k2 bin of
            kernel[i, d] is k1 bin + d - delta_k_max. Entries outside the k range are 0.
        Nmodes : number of modes of every k bin (all bins, not only k1_bins).
        nsamples : number of sampled k1 modes in each computed bin.
        meta : dict of the settings, saved with the kernel.
    """
    if isinstance(spec, str):
        spec = get_kernel(spec)
    if (windows_dir is None) == (moments is None):
        raise ValueError("Give exactly one of windows_dir and moments.")
    if windows_dir is not None:
        # The mesh is fixed by the prepare stage; read it rather than trusting the caller
        with open(os.path.join(windows_dir, "windows.json")) as f:
            wmeta = json.load(f)
        boxsize, nmesh = wmeta["boxsize"], wmeta["nmesh"]
        if max_order > wmeta["max_order"]:
            raise ValueError(
                f"max_order = {max_order} but the windows only have moments up to {wmeta['max_order']}."
            )
    if boxsize is None or nmesh is None:
        raise ValueError("boxsize and nmesh are needed with moments in memory.")
    if delta_k_max is None:
        delta_k_max = nmesh // 2 - 1

    # Same sampling as SurveyGeometry.compute_window_kernels. Every task samples all the bins
    # (cheap) with the same seed, so a k1 bin gets the same modes whichever task computes it.
    if sampled is None:
        kmodes, Nmodes = math_mc.sample_kmodes(
            kmin=kmin,
            kmax=kmax,
            dk=dk,
            boxsize=boxsize,
            max_modes=kmodes_sampled,
            k_shell_approx=k_shell_approx,
            rng=seed,
        )
    else:
        kmodes, Nmodes = sampled
    kbins = len(kmodes)
    start, stop = (0, kbins) if k1_bins is None else k1_bins
    if not 0 <= start < stop <= kbins:
        raise ValueError(f"k1_bins = {k1_bins} outside the {kbins} bins.")

    params = {
        "boxsize": float(boxsize),
        "nmesh": int(nmesh),
        "dk": float(dk),
        "delta_k_max": int(delta_k_max),
        "max_order": int(max_order),
    }

    # Tasks: each bin is split in nthreads chunks of k1 modes, as in SurveyGeometry
    tasks = []
    for i in range(start, stop):
        # Index of the bin counted from k = 0, which is what floor(|k2| / dk) gives for k2
        # (as in SurveyGeometry; kmin // dk is the number of bins below kmin)
        k1_bin_index = i + kmin // dk
        for chunk in np.array_split(kmodes[i], max(1, nthreads)):
            if len(
                chunk
            ):  # array_split gives empty chunks if there are fewer modes than threads
                tasks.append((i - start, k1_bin_index, chunk))

    ndelta = 2 * delta_k_max + 1
    kernel = np.zeros((stop - start, ndelta) + spec.shape)
    t0 = time.time()

    # Set up the workers. _kernel_chunk reads the spec, settings and moments from _STATE.
    if moments is not None:
        # Moments already in memory: put them in _STATE of this process. With fork, the worker
        # processes start as copies of this one and see _STATE without copying the arrays.
        _STATE.update(
            spec=spec, params=params, moments=dict(moments), ikgrid=_ikgrid(nmesh)
        )
        if nthreads > 1:
            pool = mp.get_context("fork").Pool(nthreads)  # workers inherit _STATE
        else:
            pool = None
    else:
        # Moments on disk: each worker memory-maps them in _init_worker. spawn starts clean
        # processes, which is safe whatever threads the parent has.
        pool = None
        if nthreads > 1:
            pool = mp.get_context("spawn").Pool(
                nthreads,
                initializer=_init_worker,
                initargs=(windows_dir, spec.to_dict(), params),
            )
        else:
            _init_worker(
                windows_dir, spec.to_dict(), params
            )  # serial: load in this process

    try:
        # imap_unordered returns results as soon as each chunk finishes; the order does not
        # matter since the chunks of a bin are added up
        results = (
            map(_kernel_chunk, tasks)
            if pool is None
            else pool.imap_unordered(_kernel_chunk, tasks)
        )
        for ibin, acc in results:
            kernel[ibin] += acc
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _STATE.clear()  # free the moments (and memory maps) of this process

    # Normalisation, as in SurveyGeometry.compute_window_kernels: average over the sampled k1
    # modes, and divide the sum over the k2 grid modes by the number of modes of the k2 bin.
    nsamples = np.array([len(kmodes[i]) for i in range(start, stop)])
    for ibin, i in enumerate(range(start, stop)):
        kernel[ibin] /= nsamples[ibin]  # mean over the sampled k1 modes
        for d in range(ndelta):
            j = i + d - delta_k_max  # index of the k2 bin among the covariance bins
            if 0 <= j < kbins:
                kernel[ibin, d] /= Nmodes[
                    j
                ]  # sum over the k2 modes -> mean over the k2 shell
            else:
                # k2 bin below kmin or above kmax: not part of the covariance
                kernel[ibin, d] = 0.0

    logger.info(f"{spec.name}, k1 bins {start}-{stop}: {time.time() - t0:.1f} s.")

    meta = {
        "spec": spec.to_dict(),
        "kmin": kmin,
        "kmax": kmax,
        "dk": dk,
        "kbins": kbins,
        "k1_bins": [start, stop],
        "kmodes_sampled": kmodes_sampled,
        "seed": seed,
        "k_shell_approx": k_shell_approx,
        "phase": "-(k1-k2)",
        **params,
    }
    return {
        "kernel": kernel,
        "Nmodes": np.asarray(Nmodes),
        "nsamples": nsamples,
        "meta": meta,
    }


def kernel_filename(name: str, start: int, stop: int) -> str:
    return f"{name}_k1bins_{start:03d}-{stop:03d}.npz"


def save_kernel(result: dict, out_dir: str) -> str:
    """Saves the output of compute_kernel to <out_dir>/<name>_k1bins_<start>-<stop>.npz."""
    os.makedirs(out_dir, exist_ok=True)
    start, stop = result["meta"]["k1_bins"]
    path = os.path.join(
        out_dir, kernel_filename(result["meta"]["spec"]["name"], start, stop)
    )
    np.savez(
        path,
        kernel=result["kernel"],
        Nmodes=result["Nmodes"],
        nsamples=result["nsamples"],
        meta=json.dumps(result["meta"]),
    )
    return path


def load_kernel(path: str) -> dict:
    with np.load(path) as f:
        return {
            "kernel": f["kernel"],
            "Nmodes": f["Nmodes"],
            "nsamples": f["nsamples"],
            "meta": json.loads(str(f["meta"])),
        }


def merge_kernel_files(out_dir: str, name: str, remove_parts: bool = False) -> str:
    """Merge stage: concatenates the k1-bin chunks of one kernel type into <out_dir>/<name>.npz.

    Checks that the chunks were computed with the same settings and cover every k1 bin once.
    """
    paths = sorted(glob.glob(os.path.join(out_dir, f"{name}_k1bins_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No files for kernel {name} in {out_dir}.")
    parts = [load_kernel(p) for p in paths]
    # Order the chunks by their first k1 bin (the zero-padded file names already give this order)
    parts.sort(key=lambda r: r["meta"]["k1_bins"][0])

    # All chunks must have the same settings (k bins, seed, mesh, max_order, ...), i.e. the same
    # metadata apart from their own k1 bin range
    ref = {k: v for k, v in parts[0]["meta"].items() if k != "k1_bins"}
    expected = 0  # first bin the next chunk should start at
    for part, path in zip(parts, paths):
        meta = {k: v for k, v in part["meta"].items() if k != "k1_bins"}
        if meta != ref:
            raise ValueError(f"{path} was computed with different settings.")
        start, stop = part["meta"]["k1_bins"]
        # Consecutive chunks must join up: [0, a), [a, b), ... with no gap or overlap
        if start != expected:
            raise ValueError(
                f"k1 bins of {name} are missing or overlap at bin {expected}."
            )
        expected = stop
    if expected != ref["kbins"]:  # the last chunk must end at the last bin
        raise ValueError(f"k1 bins {expected}-{ref['kbins']} of {name} are missing.")

    merged = {
        "kernel": np.concatenate([p["kernel"] for p in parts]),
        "Nmodes": parts[0]["Nmodes"],
        "nsamples": np.concatenate([p["nsamples"] for p in parts]),
        "meta": {**ref, "k1_bins": [0, ref["kbins"]]},
    }
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez(
        path,
        kernel=merged["kernel"],
        Nmodes=merged["Nmodes"],
        nsamples=merged["nsamples"],
        meta=json.dumps(merged["meta"]),
    )
    if remove_parts:
        for p in paths:
            os.remove(p)
    return path


# =================================================================================================
# 6. Comparison with the CovaPT layout of SurveyGeometry
# =================================================================================================

COVAPT_ELL_PAIRS = ((0, 0), (2, 2), (4, 4), (2, 0), (4, 0), (4, 2))


def to_covapt_winkernel(
    gaussian: np.ndarray, mixed: np.ndarray, shotnoise: np.ndarray
) -> np.ndarray:
    """Rebuilds SurveyGeometry.WinKernel ([kbins, 2 dkmax + 1, 15, 6]) from the kernels of an auto
    block (G_gg_gg, M_gg_g12, N_g12_g12 for gggg; the pp ones for pppp).

    Shows how the CovaPT terms map onto the kernels here (for the even multipoles 0, 2, 4):
      terms 0-8,  3a + b:  P_{2a}(k1) P_{2b}(k2):   (2l1+1)(2l2+1) [config 0 + config 1]
      term 9:     (P0(k1) + P0(k2))/2:              (2l1+1)(2l2+1) sum_p M(l'=0)
      terms 10-13: P2(k1), P4(k1), P2(k2), P4(k2):  (2l1+1)(2l2+1)/2 sum_p M(l', which)
      term 14:    constant:                         (2l1+1)(2l2+1) [N(p=(1,1)) + N(p=(1,2))]
    CovaPT multiplies by ell_factor = (2l1+1)(2l2+1) x (2 if l1 or l2 = 0) but computes only one
    of the two equal configurations / placements when a multipole is 0, which gives the same.
    """
    kb, nd = gaussian.shape[:2]
    out = np.zeros((kb, nd, 15, 6))
    for pair, (l1, l2) in enumerate(COVAPT_ELL_PAIRS):
        # Array positions of l1, l2 in ELLS_EVEN = (0, 2, 4)
        i1, i2 = l1 // 2, l2 // 2
        f = (2 * l1 + 1) * (2 * l2 + 1)
        # Terms 0-8: a, b are the positions of l'1 (spectrum at k1) and l'2 (at k2); summing the
        # last axis (config 0 + config 1) gives the two-term bracket of eq. 1.11
        for a in range(3):
            for b in range(3):
                out[..., 3 * a + b, pair] = f * gaussian[:, :, i1, i2, a, b, :].sum(
                    axis=-1
                )
        # Term 9: l' = 0. CovaPT multiplies it by (P0(k1) + P0(k2))/2, which is the 1/2 of both
        # (2-1) and (2-2) since their l' = 0 kernels are equal; so no 1/2 here. Summing the last
        # axis adds the four placements.
        out[..., 9, pair] = f * mixed[:, :, i1, i2, 0, 0, :].sum(axis=-1)
        # Terms 10-13: (position of l', which): l' = 2, 4 with P(k1) (which 0), then with P(k2)
        for term, (j, which) in zip((10, 11, 12, 13), ((1, 0), (2, 0), (1, 1), (2, 1))):
            out[..., term, pair] = f / 2 * mixed[:, :, i1, i2, j, which, :].sum(axis=-1)
        # Term 14: eq. 2.31 has L_l1 at x1 only, i.e. placements (1, 1) and (1, 2)
        out[..., 14, pair] = f * (
            shotnoise[:, :, i1, i2, 0] + shotnoise[:, :, i1, i2, 1]
        )
    return out


# =================================================================================================
# 7. Command line (job array)
# =================================================================================================


def _tasks(names: list[str], n_k1_chunks: int) -> list[tuple[str, int]]:
    """All (kernel name, k1 chunk) pairs. Task i of the job array is element i of this list,
    i.e. kernel i // n_k1_chunks and chunk i % n_k1_chunks."""
    return [(name, c) for name in names for c in range(n_k1_chunks)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # type: ignore
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser(
        "list", help="List the kernel types and the job array tasks."
    )
    p_list.add_argument("--n-k1-chunks", type=int, default=1)
    p_list.add_argument(
        "--block",
        default=None,
        help="Only list the kernel types (and tasks) of this block, e.g. pgpg.",
    )

    p_comp = sub.add_parser(
        "compute",
        help="Compute the kernels of one covariance block, one (kernel type, k1 chunk) per task.",
    )
    p_comp.add_argument(
        "--block",
        required=True,
        help=f"Covariance block, one of {list(BLOCKS)}. One block per job array.",
    )
    p_comp.add_argument("--windows-dir", required=True)
    p_comp.add_argument("--out-dir", required=True)
    p_comp.add_argument("--kmin", type=float, required=True)
    p_comp.add_argument("--kmax", type=float, required=True)
    p_comp.add_argument("--dk", type=float, required=True)
    p_comp.add_argument(
        "--kernels",
        default=None,
        help="Comma-separated subset of the kernel types of the block, e.g. to rerun a failed "
        "one. Default: all the kernel types of the block.",
    )
    p_comp.add_argument(
        "--n-k1-chunks",
        type=int,
        default=1,
        help="Number of chunks the k1 bins are split into (tasks per kernel type).",
    )
    p_comp.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Task index: kernel = id // n-k1-chunks, chunk = id %% n-k1-chunks. "
        "Default: $SLURM_ARRAY_TASK_ID, or run all tasks.",
    )
    p_comp.add_argument("--kmodes-sampled", type=int, default=5000)
    p_comp.add_argument("--max-order", type=int, default=4)
    p_comp.add_argument("--delta-k-max", type=int, default=None)
    p_comp.add_argument(
        "--nthreads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 1))
    )
    p_comp.add_argument("--seed", type=int, default=42)

    p_merge = sub.add_parser("merge", help="Merge the k1 chunks of every kernel type.")
    p_merge.add_argument("--out-dir", required=True)
    p_merge.add_argument("--remove-parts", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.command == "list":
        if args.block is not None:
            check_block(args.block)
        blocks = list(BLOCKS) if args.block is None else [args.block]
        for block in blocks:
            tasks = _tasks(list(BLOCKS[block]), args.n_k1_chunks)
            print(f"{block}:")
            for name in BLOCKS[block]:
                spec = get_kernel(name)
                print(
                    f"  {spec.name:14s} {spec.kind:9s} x1: {spec.window1:4s} x2: {spec.window2:4s} "
                    f"shape {spec.shape}  ({spec.used_in})"
                )
            print(
                f"  {len(tasks)} tasks with --n-k1-chunks {args.n_k1_chunks} "
                f"(#SBATCH --array=0-{len(tasks) - 1})\n"
            )
        return

    if args.command == "merge":
        # Kernel names from the chunk files present, e.g. "G_pg_gg_k1bins_000-005.npz" -> "G_pg_gg"
        names = sorted(
            {
                os.path.basename(p).split("_k1bins_")[0]
                for p in glob.glob(os.path.join(args.out_dir, "*_k1bins_*.npz"))
            }
        )
        for name in names:
            print(
                merge_kernel_files(args.out_dir, name, remove_parts=args.remove_parts)
            )
        return

    # "compute": work out which (kernel, chunk) tasks this process runs. The block fixes the
    # kernel types, so every task of the job array belongs to that one block.
    check_block(args.block)
    names = list(BLOCKS[args.block])
    if args.kernels is not None:
        subset = args.kernels.split(",")
        wrong = [n for n in subset if n not in names]
        if wrong:
            raise ValueError(
                f"Kernel types {wrong} are not part of block {args.block}, which has {names}."
            )
        names = subset
    tasks = _tasks(names, args.n_k1_chunks)
    task_id = args.task_id
    if task_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id is not None and not 0 <= task_id < len(tasks):
        raise ValueError(
            f"Task id {task_id} out of range: block {args.block} has {len(tasks)} tasks "
            f"(see 'python -m thecov.kernels list --block {args.block}')."
        )
    # Without a task id (e.g. on a laptop) run every task one after the other
    selected = tasks if task_id is None else [tasks[task_id]]

    # Number of k bins, to split them into chunks. sample_kmodes is only called to count the bins
    # (Nmodes), so one mode per bin and any seed will do.
    with open(os.path.join(args.windows_dir, "windows.json")) as f:
        boxsize = json.load(f)["boxsize"]
    _, Nmodes = math_mc.sample_kmodes(
        kmin=args.kmin,
        kmax=args.kmax,
        dk=args.dk,
        boxsize=boxsize,
        max_modes=1,
        k_shell_approx=0.1,
        rng=0,
    )
    # Contiguous chunks of bin indices, as equal in size as possible
    chunks = np.array_split(np.arange(len(Nmodes)), args.n_k1_chunks)

    for name, c in selected:
        if len(chunks[c]) == 0:  # more chunks than bins
            continue
        result = compute_kernel(
            name,
            args.kmin,
            args.kmax,
            args.dk,
            windows_dir=args.windows_dir,
            k1_bins=(int(chunks[c][0]), int(chunks[c][-1]) + 1),
            kmodes_sampled=args.kmodes_sampled,
            max_order=args.max_order,
            delta_k_max=args.delta_k_max,
            nthreads=args.nthreads,
            seed=args.seed,
        )
        print(save_kernel(result, args.out_dir))


if __name__ == "__main__":
    main()
