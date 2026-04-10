"""Module containing basic classes to deal with covariance matrices."""

import os
import itertools as itt
import copy

import numpy as np

from . import utils, math, geometry
import logging
from typing import Any, Optional

__all__ = [
    "Covariance",
    "MultipoleCovariance",
    "FourierBinned",
    "FourierCovariance",
    "MultipoleFourierCovariance",
    "PowerSpectrumMultipolesCovariance",
]


class Covariance:
    """A class that represents a covariance matrix.
    Implements basic operations such as correlation matrix computation, etc.
    """

    def __init__(self, covariance: None | np.ndarray = None):
        """Initializes a Covariance object.

        Parameters
        ----------
        covariance : numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.
        """

        self._covariance = covariance

    @property
    def cov(self):
        """The covariance matrix.

        Returns
        -------
        numpy.ndarray
            (n,n) numpy array corresponding to the elements of the covariance matrix.
        """

        return self._covariance

    @cov.setter
    def cov(self, covariance: None | np.ndarray):
        """Sets the covariance matrix.

        Parameters
        ----------
        covariance : numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.
        """

        self._covariance = covariance

    @property
    def cor(self) -> np.ndarray:
        """Returns the correlation matrix.

        The correlation matrix is obtained by dividing each element of the covariance matrix by
        the product of the standard deviations of the corresponding variables.

        Returns
        -------
        numpy.ndarray
            (n,n) numpy array corresponding to the elements of the correlation matrix.
        """

        cov = self.cov
        # Prevent further calculations if covariance matrix is not set.
        assert cov is not None, "Covariance matrix is not set."
        v = np.sqrt(np.diag(cov))
        outer_v = np.outer(v, v)
        outer_v[outer_v == 0] = np.inf
        cor = cov / outer_v
        cor[cov == 0] = 0
        return cor

    def symmetrize(self):
        """Symmetrizes the covariance matrix in place."""
        assert self.cov is not None, "Covariance matrix is not set."
        self.cov = (self.cov + self.cov.T) / 2

    def symmetrized(self) -> "Covariance":
        """Returns a symmetrized copy of the covariance matrix.

        Returns
        -------
        Covariance
            Covariance object corresponding to the symmetrized covariance matrix.
        """
        new_cov = copy.deepcopy(self)
        new_cov.symmetrize()
        return new_cov

    def regularize(self, mode: str = "zero"):
        """Regularizes the covariance matrix in place by modifying the eigenvalues.

        Parameters
        ----------
        mode : str
            The mode of regularization. Supported modes are "zero", "flip", "minpos", and "nearest".
            "zero": sets negative eigenvalues to zero.
            "flip": takes the absolute value of the eigenvalues.
            "minpos": sets negative eigenvalues to the minimum positive eigenvalue.
            "nearest": computes the nearest positive definite matrix using the Higham 1988 algorithm https://doi.org/10.1016/0024-3795(88)90223-6. See stackoverflow discussion for implementation details: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
        """
        eigvals, eigvecs = self.eig
        if mode == "zero":
            eigvals[eigvals < 0] = 0
        elif mode == "flip":
            eigvals = np.abs(eigvals)
        elif mode == "minpos":
            eigvals[eigvals < 0] = min(eigvals[eigvals > 0])
        elif mode == "nearest":
            assert self.cov is not None, "Covariance matrix is not set."
            # Computing the nearest symmetric positive definite matrix using the Higham 1988 algorithm https://doi.org/10.1016/0024-3795(88)90223-6.
            # See stackoverflow discussion for implementation details: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
            B = (self.cov + self.cov.T) / 2
            _, S, V = np.linalg.svd(B)
            H = np.dot(V.T * S, V)
            A2 = (B + H) / 2
            A3 = (A2 + A2.T) / 2
            try:
                _ = np.linalg.cholesky(A3)
                isPD = True
            except np.linalg.LinAlgError:
                isPD = False
            if isPD:
                self.cov = A3
            else:
                spacing = np.spacing(np.linalg.norm(self.cov))
                I = np.eye(self.cov.shape[0])
                k = 1
                while not isPD:
                    min_eig = np.min(np.real(np.linalg.eigvals(A3)))
                    A3 += I * (-min_eig * k**2 + spacing)
                    try:
                        _ = np.linalg.cholesky(A3)
                        isPD = True
                    except np.linalg.LinAlgError:
                        isPD = False
                    k += 1

                self.cov = A3
        else:
            raise ValueError(
                f"Mode {mode} not recognized. Supported modes are zero, flip, minpos, nearest."
            )

        if mode in ["zero", "flip", "minpos"]:
            self.cov = np.einsum("ij,jk,kl->il", eigvecs, np.diag(eigvals), eigvecs.T)

    def regularized(self) -> "Covariance":
        """Returns a regularized copy of the covariance matrix."""
        new_cov = copy.deepcopy(self)
        new_cov.regularize()
        return new_cov

    def __add__(self, y: "Covariance |float|int") -> "Covariance":
        """Adds another covariance matrix or a scalar to the covariance matrix.

        Parameters
        ----------
        y : Covariance or scalar
            The covariance matrix or scalar to add.

        Returns
        -------
        Covariance
            The resulting covariance matrix after addition.
        """
        assert self.cov is not None, "Covariance matrix is not set."
        return Covariance(self.cov + (y.cov if isinstance(y, Covariance) else y))

    def __neg__(self) -> "Covariance":
        """Returns a new Covariance instance with the sign of the matrix flipped."""
        assert self.cov is not None, "Covariance matrix is not set."
        return Covariance(-self.cov)

    def __sub__(self, y: "Covariance |float|int") -> "Covariance":
        """Subtracts another covariance matrix or a scalar from the covariance matrix.

        Parameters
        ----------
        y : Covariance or scalar
            The covariance matrix or scalar to subtract.

        Returns
        -------
        Covariance
            The resulting covariance matrix after subtraction.
        """
        return self.__add__(-y)

    def __mul__(self, y: "float|int") -> "Covariance":
        """Multiplies the covariance matrix by a scalar.

        Parameters
        ----------
        y : scalar
            The scalar to multiply with the covariance matrix.

        Returns
        -------
        Covariance
            The resulting covariance matrix after multiplication.
        """
        assert self.cov is not None, "Covariance matrix is not set."
        return Covariance(self.cov * y)

    def __truediv__(self, y: "float|int") -> "Covariance":
        """Divides the covariance matrix by a scalar.

        Parameters
        ----------
        y : scalar
            The scalar to divide the covariance matrix by.

        Returns
        -------
        Covariance
            The resulting covariance matrix after division.
        """
        assert self.cov is not None, "Covariance matrix is not set."
        return Covariance(self.cov / y)

    @property
    def T(self) -> "Covariance":
        """Returns the transpose of the covariance matrix.

        Returns
        -------
        Covariance
            Covariance object corresponding to the transpose of the covariance matrix.
        """

        obj = copy.deepcopy(self)
        assert obj.cov is not None, "Covariance matrix is not set."
        obj.cov = obj.cov.T

        return obj

    @property
    def shape(self) -> tuple:
        """Returns the shape of the covariance.

        Returns
        -------
        tuple
            A tuple with the shape of the covariance matrix.
        """

        assert self.cov is not None, "Covariance matrix is not set."

        return self.cov.shape

    @property
    def eig(self) -> tuple:
        """Compute the eigenvalues and right eigenvectors of the covariance.

        Returns
        -------
        A namedtuple with the following attributes:
            eigenvalues
            (..., M) array
                The eigenvalues, each repeated according to its multiplicity.
                The eigenvalues are not necessarily ordered. The resulting
                array will be of complex type, unless the imaginary part is
                zero in which case it will be cast to a real type. When a is
                real the resulting eigenvalues will be real (0 imaginary
                part) or occur in conjugate pairs

            eigenvectors
            (...), M, M) array
                The normalized (unit “length”) eigenvectors, such that the
                column eigenvectors[:,i] is the eigenvector corresponding to
                the eigenvalue eigenvalues[i].
        """

        assert self.cov is not None, "Covariance matrix is not set."
        return np.linalg.eig(self.cov)

    @property
    def eigvals(self) -> np.ndarray:
        """Compute the eigenvalues of the covariance.

        Returns
        -------
        (..., M,) ndarray
            The eigenvalues, each repeated according to its multiplicity.
            They are not necessarily ordered, nor are they necessarily
            real for real matrices.
        """

        assert self.cov is not None, "Covariance matrix is not set."
        return np.linalg.eigvals(self.cov)

    def save(self, filename: str):
        """Saves the covariance as a .npz file with a specified filename.

        Parameters
        -------
        filename : string
            The name of the file where the covariance matrix will be saved.
        """
        utils.mkdir(os.path.dirname(filename))
        assert self.cov is not None, "Covariance matrix is not set."
        np.savez(
            filename
            if filename.strip()[-4:] in (".npz", ".npy")
            else f"{filename}.npz",
            covariance=self.cov,
        )

    def savetxt(self, filename: str):
        """Saves the covariance as a text file with a specified filename.

        Parameters
        -------
        filename : string
            The name of the file where the covariance matrix will be saved.
        """
        utils.mkdir(os.path.dirname(filename))
        assert self.cov is not None, "Covariance matrix is not set."
        np.savetxt(filename, self.cov)

    @classmethod
    def load(cls: type["Covariance"], filename: str) -> "Covariance":
        """Loads the covariance from a .npz file with a specified filename.

        Parameters
        -------
        cls: type["Covariance"]
        filename : string
            The name of the file where the covariance matrix will be loaded from.
        """

        with np.load(filename, mmap_mode="r") as data:
            return cls(data["covariance"])

    @classmethod
    def loadtxt(cls: type["Covariance"], *args: Any, **kwargs: Any) -> "Covariance":
        """Loads the covariance from a text file with a specified filename.

        Parameters
        -------
        *args
            Arguments to be passed to numpy.loadtxt.
        **kwargs
            Keyword arguments to be passed to numpy.loadtxt.

        Returns
        -------
        Covariance
            Covariance object.
        """

        return cls.from_array(np.loadtxt(*args, **kwargs))

    @classmethod
    def from_array(cls: type["Covariance"], a: np.ndarray) -> "Covariance":
        """Creates a Covariance object from a numpy array.

        Parameters
        -------
        a: np.ndarray
            (n,n) numpy array with elements corresponding to the covariance.

        Returns
        -------
        Covariance
            Covariance object.
        """

        return cls(covariance=a)


class MultipoleCovariance(Covariance):
    """A class to represent a covariance matrix for a set of multipoles.

    Attributes
    ----------
    cov : numpy.ndarray
        The covariance matrix.
    cor : numpy.ndarray
        The correlation matrix.
    """

    def __init__(self):
        # Initializes an empty MultipoleCovariance object. The covariance matrix is stored as a dictionary of covariance matrices for different multipole pairs,
        # and the full covariance matrix is obtained by stacking these matrices.
        self._multipole_covariance = {}
        self._ells = []
        self._mshape = (0, 0)

    def __add__(
        self, y: "MultipoleCovariance |Covariance |float|int"
    ) -> "MultipoleCovariance":
        """Adds another covariance matrix or a scalar to the covariance matrix."""
        assert self.cov is not None, "Multipole covariance matrix is not set."
        if isinstance(y, MultipoleCovariance):
            assert self.ells == y.ells, "ells are not the same"

        # return MultipoleCovariance.from_array(
        #     self.cov + (y.cov if isinstance(y, Covariance) else y), self.ells
        # )
        if isinstance(y, Covariance):
            y_val = y.cov
            assert y_val is not None, "Input covariance matrix is not set."

        else:
            y_val = y

        return MultipoleCovariance.from_array(self.cov + y_val, self.ells)

    def __sub__(
        self, y: "MultipoleCovariance |Covariance |float|int"
    ) -> "MultipoleCovariance":
        """Subtracts another covariance matrix or a scalar from the covariance matrix."""
        return self.__add__(-y)

    def __mul__(self, y: float | int) -> "MultipoleCovariance":
        """Multiplies the covariance matrix by a scalar."""
        return MultipoleCovariance.from_array(self.cov * y, self.ells)

    def __truediv__(self, y: float | int) -> "MultipoleCovariance":
        """Divides the covariance matrix by a scalar."""
        return MultipoleCovariance.from_array(self.cov / y, self.ells)

    @property
    def cov(self) -> np.ndarray:
        """This function calculates the full covariance matrix by stacking covariances for different multipoles
        in ascending order.

        Returns
        -------
        numpy.ndarray
            An (n,n) numpy array corresponding to the elements of the covariance matrix.
        """

        ells = self.ells
        full_cov = np.zeros(np.array(self._mshape) * len(ells))
        for (i, l1), (j, l2) in itt.product(enumerate(ells), enumerate(ells)):
            cov_ell = self.get_ell_cov(l1, l2)
            assert cov_ell is not None, (
                f"Covariance for multipoles {l1} and {l2} is not set."
            )
            full_cov[
                i * self._mshape[0] : (i + 1) * self._mshape[0],
                j * self._mshape[1] : (j + 1) * self._mshape[1],
            ] = cov_ell.cov
        return full_cov

    @cov.setter
    def cov(self, covariance: np.ndarray | None):
        """Sets the full covariance matrix from covariances for different multipoles stacked
        in ascending order.

        Parameters
        ----------
        covariance : np.ndarray | None
            An (n,n) numpy array corresponding to the elements of the covariance matrix.
        """

        assert covariance is not None, "Covariance matrix cannot be set to None."

        self.set_full_cov(covariance, self.ells)

    @property
    def ells(self) -> tuple[int, ...] | list[int]:
        """The multipoles for which the covariance matrix is defined. Sorted in ascending order.

        Returns
        -------
        tuple
            A tuple of multipole values.
        """

        return sorted(self._ells)

    def get_ell_cov(
        self, l1: int, l2: int, force_return: bool | float = False, cls=Covariance
    ) -> Optional[Covariance]:
        """Returns the covariance matrix for a given pair of multipoles.

        Parameters
        ----------
        l1
            the first multipole.
        l2
            the second multipole.
        force_return, boolean, float, optional
            If True, returns a zero matrix if the covariance matrix is not defined.
            If `force_return` is a float, returns a matrix with the given value.

        Returns
        -------
        Covariance
            A Covariance object corresponding to the covariance matrix for the given multipoles.
        """

        if l1 > l2:
            cov_ell = self.get_ell_cov(l2, l1)
            assert cov_ell is not None, (
                f"Covariance for multipoles {l2} and {l1} is not set, so covariance for {l1} and {l2} cannot be obtained by transposition."
            )
            return cov_ell.T

        if (l1, l2) in self._multipole_covariance:
            return self._multipole_covariance[l1, l2]
        elif type(force_return) is not bool:
            return cls(force_return * np.ones(self._mshape))
        elif force_return:
            return cls(np.zeros(self._mshape))

    def set_ell_cov(
        self, l1: int, l2: int, cov: Covariance | np.ndarray, cls=Covariance
    ) -> Covariance:
        """Sets the covariance matrix for a given pair of multipoles.

        Parameters
        ----------
        l1 : int
            The first multipole.
        l2 : int
            The second multipole.
        cov : Covariance or numpy.ndarray
            The covariance matrix. Can be an instance of Covariance or a numpy array.
        """

        if l1 > l2:
            return self.set_ell_cov(l2, l1, cov.T)

        if self._ells == []:
            self._mshape = cov.shape

        # assert cov.shape == self._mshape, "ell covariance has shape inconsistent with other ells"

        # Conversion to list to allow appending if ells is currently a tuple.
        # This allows users to set covariances for different multipole pairs in any order without having to specify all the multipoles at the beginning.
        if isinstance(self._ells, tuple):
            self._ells = list(self._ells)

        if l1 not in self.ells:
            self._ells.append(l1)
        if l2 not in self.ells:
            self._ells.append(l2)

        cov = cov if isinstance(cov, cls) else cls(cov)

        self._multipole_covariance[l1, l2] = cov

        return cov

    def set_full_cov(
        self, cov_array: np.ndarray, ells: tuple[int, ...] | list[int] = (0, 2, 4)
    ) -> "MultipoleCovariance":
        """Sets the full covariance matrix from stacked covariances for different multipoles.

        Parameters
        ----------
        cov_array
            (n,n) numpy array with elements corresponding to the covariance.
        ells
            the multipoles for which the covariance matrix is defined.

        Returns
        -------
        MultipoleCovariance
            A MultipoleCovariance object.
        """

        assert cov_array.ndim == 2, "Covariance should be a matrix (ndim == 2)."
        assert cov_array.shape[0] == cov_array.shape[1], (
            "Covariance matrix should be a square matrix."
        )
        assert cov_array.shape[0] % len(ells) == 0, (
            "Covariance matrix shape should be a multiple of the number of ells."
        )

        c = cov_array
        self._ells = ells
        self._mshape = tuple(np.array(cov_array.shape) // len(ells))

        for (i, l1), (j, l2) in itt.combinations_with_replacement(enumerate(ells), r=2):
            self.set_ell_cov(
                l1,
                l2,
                c[
                    i * c.shape[0] // len(ells) : (i + 1) * c.shape[0] // len(ells),
                    j * c.shape[1] // len(ells) : (j + 1) * c.shape[1] // len(ells),
                ],
            )
        return self

    def foreach(self, func: Any) -> "MultipoleCovariance":
        """Applies a function to each covariance matrix.

        Parameters
        ----------
        func : function
            The function to be applied to each covariance matrix.
        """

        for (l1, l2), cov in self._multipole_covariance.items():
            self.set_ell_cov(l1, l2, func(cov))

        return self

    @classmethod
    def from_array(cls, *args: Any, **kwargs: Any) -> "MultipoleCovariance":
        """Creates a MultipoleCovariance object from a numpy array corresponding to the full covariance matrix.

        Parameters
        ----------
        cov_array
            (n,n) numpy array with elements corresponding to the covariance.
        ells
            the multipoles for which the covariance matrix is defined.

        Returns
        -------
        MultipoleCovariance
            A MultipoleCovariance object.
        """

        cov = cls()
        cov.set_full_cov(*args, **kwargs)

        return cov

    def loadtxt(self, *args: Any, **kwargs: Any) -> "MultipoleCovariance":
        """Loads the covariance from a text file with a specified filename.

        Parameters
        ----------
        filename
            The name of the file where the covariance matrix will be loaded from.

        Returns
        -------
        MultipoleCovariance
            A MultipoleCovariance object.
        """

        self.set_full_cov(np.loadtxt(*args, **kwargs))
        return self


class FourierBinned:
    """A class to represent a power spectrum binned in wavenumber k. Only linear binning is supported.

    Attributes
    ----------
    kmin: float
        The minimum value of the wavenumber k.
    kmax: float
        The maximum value of the wavenumber k.
    dk: float
        The spacing between k-bins.
    nmodes: numpy.ndarray, optional
        The number of modes to be used in the calculation. It is an optional parameter.
        If omitted, it is calculated from the volume of spherical shells.

    Methods
    -------
    set_kbins
        This function defines the k-bins. Only linear binning is supported.
    """

    def __init__(self) -> None:
        self.kmin, self.kmax, self.dk, self._nmodes = None, None, None, None

    def set_kbins(
        self, kmin: float, kmax: float, dk: float, nmodes: np.ndarray | None = None
    ) -> None:
        """This function defines the k-bins. Only linear binning is supported.

        Parameters
        ----------
        kmin: float
            The minimum value of the wavenumber k.
        kmax: float
            The maximum value of the wavenumber k.
        dk: float
            The spacing between k-bins.
        nmodes: numpy.ndarray, optional
            The number of modes to be used in the calculation. It is an optional parameter.
            If omitted, it is calculated from the volume of spherical shells.
        """

        self.dk = dk
        self.kmax = kmax
        self.kmin = kmin

        self._nmodes = nmodes

    @property
    def is_kbins_set(self) -> bool:
        """Check if k-bins were defined.

        Returns
        -------
            bool, True if k-bins were defined, False otherwise.
        """
        if hasattr(self, "dk") and hasattr(self, "kmin") and hasattr(self, "kmax"):
            return None not in (self.dk, self.kmin, self.kmax)
        else:
            return False

    @property
    def kbins(self) -> int:
        """Returns the total number of k-bins.

        Returns
        -------
        int
            The total number of k-bins.
        """

        return len(self.kmid)

    @property
    def kmid(self) -> np.ndarray:
        """
        Returns the midpoints of the k-bins.

        Returns
        -------
        numpy.ndarray
            The midpoints of the k-bins.
        """
        assert (
            self.kmin is not None and self.kmax is not None and self.dk is not None
        ), (
            "k-bins are not set. Please set kmin, kmax, and dk using the set_kbins method."
        )

        return np.arange(self.kmin + self.dk / 2, self.kmax + self.dk / 2, self.dk)

    @property
    def kavg(self) -> np.ndarray:
        """
        Returns the average k of the k-bins. Assumes spherical approximation to
        integrate k-modes, which fails for small k.

        Returns
        -------
        numpy.ndarray
            The average k of the k-bins.
        """
        return (
            3
            / 4
            * (self.kedges[1:] ** 4 - self.kedges[:-1] ** 4)
            / (self.kedges[1:] ** 3 - self.kedges[:-1] ** 3)
        )

    @property
    def kedges(self) -> np.ndarray:
        """
        Returns the edges of the k-bins.

        Returns
        -------
        numpy.ndarray
            The edges of the k-bins.
        """

        assert (
            self.kmin is not None and self.kmax is not None and self.dk is not None
        ), (
            "k-bins are not set. Please set kmin, kmax, and dk using the set_kbins method."
        )

        return np.arange(self.kmin, self.kmax + self.dk / 2, self.dk)

    @property
    def kfun(self):
        """Fundamental wavenumber of the box 2*pi/Lbox.

        Returns
        -------
        float
            The fundamental wavenumber of the box.
        """

        assert self.volume is not None, (
            "Volume is not set. Cannot compute fundamental wavenumber."
        )

        return 2 * np.pi / self.volume ** (1 / 3)

    @property
    def volume(self) -> Optional[float]:
        """Returns the volume of the object. If not available, return that of the associated geometry.

        Returns
        -------
        float
            The volume of the object.
        """

        # if hasattr(self, "_volume"):
        #     return self._volume

        # if hasattr(self, "geometry"):
        #     return self.geometry.volume

        # Try to get _volume, return None if it doesn't exist
        vol = getattr(self, "_volume", None)
        if vol is not None:
            return vol

        # Try to get geometry
        geom = getattr(self, "geometry", None)
        if geom is not None:
            return geom.volume

        return None

    @property
    def nmodes(self) -> np.ndarray:
        """This function calculates the number of modes per k-bin shell. If nmodes was not provided, it is
        estimated from the volume of each shell.

        Returns
        -------
        numpy.ndarray
            The number of modes per k-bin shell.
        """

        if self._nmodes is not None:
            return self._nmodes

        return math.nmodes(self.volume, self.kedges[:-1], self.kedges[1:])

    @nmodes.setter
    def nmodes(self, nmodes: np.ndarray) -> np.ndarray:
        """Manually sets the number of modes per k-bin shell.

        Parameters
        -------
        nmodes : numpy.ndarray
            The number of modes per k-bin shell.
        """

        self._nmodes = nmodes
        return nmodes


class FourierCovariance(Covariance, FourierBinned):
    def __init__(self, cov):
        Covariance.__init__(self, cov)
        FourierBinned.__init__(self)

    @property
    def kmid_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the kmid matrices for the covariance matrix."""
        k1 = np.einsum("i,j->ij", self.kmid, np.ones_like(self.kmid))
        k2 = k1.T

        return k1, k2

    @property
    def kmin_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the kmin matrices for the covariance matrix."""
        k1 = np.einsum("i,j->ij", self.kedges[:-1], np.ones_like(self.kmid))
        k2 = k1.T

        return k1, k2

    def kcut(self, kmin: Optional[float] = None, kmax: Optional[float] = None):
        """Cuts the covariance matrix to a specified kmin and kmax."""
        if kmin is None:
            kmin = self.kmin

        if kmax is None:
            kmax = self.kmax

        imin = (self.kmid >= kmin).argmax()
        imax = (
            len(self.kmid)
            if (self.kmid <= kmax).all()
            else (self.kmid <= kmax).argmin()
        )

        self._covariance = self._covariance[imin:imax, imin:imax]
        self.kmin, self.kmax = kmin, kmax

        return self


class MultipoleFourierCovariance(MultipoleCovariance, FourierCovariance):
    def __init__(self):
        MultipoleCovariance.__init__(self)
        FourierBinned.__init__(self)
        self.logger = logging.getLogger("MultipoleFourierCovariance")

    @property
    def kmid_ell_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the kmid matrices for the covariance matrix."""
        kfull = np.concatenate([self.kmid for _ in self.ells])
        k1 = np.einsum("i,j->ij", kfull, np.ones_like(kfull))
        k2 = k1.T

        return k1, k2

    @property
    def ell_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the ell matrices for the covariance matrix."""
        ell_array = np.einsum("i,j->ij", self.ells, np.ones(self.kbins)).flatten()

        ell1 = np.einsum("i,j->ij", ell_array, np.ones_like(ell_array))
        ell2 = ell1.T

        return ell1, ell2

    def savecsv(
        self,
        filename: str,
        ells_both_ways: bool = False,
        fmt: list[str] = ["%.d", "%.d", "%.4f", "%.4f", "%.8e"],
    ):
        """Saves the covariance as a text file with a specified filename.

        Parameters        ----------
        filename : string
            The name of the file where the covariance matrix will be saved.
        ells_both_ways : bool
            If True, saves the covariance for both (ell1, ell2) and (ell2, ell1) pairs. If False, saves only for ell1 <= ell2 to avoid redundancy. Default is False.
        fmt : list of str
            A list of format strings for each column in the output file. Default is ["%.d", "%.d", "%.4f", "%.4f", "%.8e"] corresponding to integer format for ell1 and ell2,
            float format with 4 decimal places for k1 and k2, and scientific notation with 8 decimal places for the covariance value.
        """
        k1, k2 = self.kmid_ell_matrices
        ell1, ell2 = self.ell_matrices

        cov = self.cov

        mask = ell1 <= ell2 if not ells_both_ways else np.ones_like(ell1, dtype=bool)
        utils.mkdir(os.path.dirname(filename))
        np.savetxt(
            filename,
            np.concatenate(
                [
                    ell1[mask].reshape(-1, 1),
                    ell2[mask].reshape(-1, 1),
                    k1[mask].reshape(-1, 1),
                    k2[mask].reshape(-1, 1),
                    cov[mask].reshape(-1, 1),
                ],
                axis=1,
            ),
            fmt=fmt,
            header="ell1 ell2 kmid1 kmid2 cov",
        )

    def loadcsv(self, filename: str) -> "MultipoleFourierCovariance":
        """Loads the covariance from a text file with a specified filename.

        Parameters
        ----------
        filename : string
            The name of the file from which the covariance matrix will be loaded.
        Returns
        -------
        MultipoleFourierCovariance
            A MultipoleFourierCovariance object.
        """
        ell1, ell2, k1, k2, value = np.loadtxt(filename).T

        k = np.unique(k1)
        kbins = len(k)

        assert np.allclose(k, np.unique(k2)), "k1 and k2 are not consistent"

        dk = np.mean(np.diff(k))
        kmin = k.min() - dk / 2
        kmax = k.max() + dk / 2

        ells = np.unique(ell1)
        assert np.allclose(ells, np.unique(ell2)), "ell1 and ell2 are not consistent"

        ells_both_ways = len(value) == (len(ells) * kbins) ** 2
        ells_one_way = len(value) == (len(ells) ** 2 + len(ells)) / 2 * kbins**2

        assert ells_one_way or ells_both_ways, (
            "length of covariance file doesn'nt match"
        )

        self.set_kbins(kmin, kmax, float(dk))

        assert np.allclose(np.unique(k1), self.kmid), "k bins are not linearly spaced"

        kmid_matrix = np.einsum("i,j->ij", k, np.ones_like(k))

        for l1, l2 in itt.combinations_with_replacement(ells, r=2):
            block_mask = (ell1 == l1) & (ell2 == l2)
            assert np.allclose(k1[block_mask].reshape(kmid_matrix.shape), kmid_matrix)
            assert np.allclose(
                k2[block_mask].reshape(kmid_matrix.T.shape), kmid_matrix.T
            )
            c = value[block_mask].reshape(kbins, kbins)
            self.set_ell_cov(l1, l2, c)

        return self

    @classmethod
    def fromcsv(cls: Any, filename: str) -> "MultipoleFourierCovariance":
        """Creates a MultipoleFourierCovariance object from a text file with a specified filename.
        Parameters
        ----------
        cls: class
        filename : string
            The name of the file from which the covariance matrix will be loaded.
        Returns
        -------
        MultipoleFourierCovariance
            A MultipoleFourierCovariance object."""
        cov = cls()
        cov.loadcsv(filename)
        return cov

    def set_ell_cov(
        self,
        l1: int,
        l2: int,
        cov: Any,
        cls: Any = FourierCovariance,
    ) -> Any:
        """Sets the covariance matrix for a given pair of multipoles.
        Parameters
        ----------
        l1 : int
            The first multipole.
        l2 : int
            The second multipole.
        cov : Covariance or numpy.ndarray
            The covariance matrix. Can be an instance of Covariance or a numpy array.
        cls : class
            The class to be used for the covariance matrix. Default is FourierCovariance.

        Returns
        -------
        Covariance
            A Covariance object corresponding to the covariance matrix for the given multipoles.
        """
        cov = super().set_ell_cov(l1, l2, cov, cls=cls)

        if not cov.is_kbins_set:
            cov.set_kbins(self.kmin, self.kmax, self.dk, self._nmodes)
        return cov

    def get_ell_cov(
        self,
        l1: int,
        l2: int,
        force_return: bool | float = False,
        cls: Any = FourierCovariance,
    ) -> Any:
        """Returns the covariance matrix for a given pair of multipoles.
        Parameters
        ----------
        l1
            the first multipole.
        l2
            the second multipole.
        force_return, boolean, float, optional
            If True, returns a zero matrix if the covariance matrix is not defined.
            If `force_return` is a float, returns a matrix with the given value.
        cls : class
            The class to be used for the covariance matrix. Default is FourierCovariance."""
        return super().get_ell_cov(l1, l2, force_return, cls)

    def kcut(
        self, kmin: float | None = None, kmax: float | None = None
    ) -> "MultipoleFourierCovariance":
        """Cuts the covariance matrix to a specified kmin and kmax. If kmin or kmax is not provided, it uses the existing kmin or kmax of the object.
        Parameters
        ----------
        kmin : float, optional
            The minimum value of the wavenumber k. If not provided, it uses the existing kmin of the object.
        kmax : float, optional
            The maximum value of the wavenumber k. If not provided, it uses the existing kmax of the object.
        Returns
        -------
        MultipoleFourierCovariance
            The object itself with the covariance matrix cut to the specified kmin and kmax.
        """
        if kmin is None:
            kmin = self.kmin

        if kmax is None:
            kmax = self.kmax

        assert kmin is not None and kmax is not None and self.dk is not None, (
            "kmin and kmax cannot both be None."
        )

        self.foreach(lambda cov: cov.kcut(kmin, kmax))
        self.set_kbins(kmin, kmax, self.dk)

        self.logger.info(f"kcut to {self.kmin} < k < {self.kmax}")

        return self

    def set_kbins(
        self, kmin: float, kmax: float, dk: float, nmodes: np.ndarray | None = None
    ) -> Any:
        """This function defines the k-bins. Only linear binning is supported.
        Parameters
        ----------
        kmin: float
            The minimum value of the wavenumber k.
        kmax: float
            The maximum value of the wavenumber k.
        dk: float
            The spacing between k-bins.
        nmodes: numpy.ndarray, optional
            The number of modes to be used in the calculation. It is an optional parameter.
            If omitted, it is calculated from the volume of spherical shells.
        Returns
        -------
        MultipoleFourierCovariance
            The object itself with the k-bins set to the specified values.
        """
        size = (kmax - kmin) / dk
        # size = (np.round(size) if np.allclose(np.round(size), size) else size).astype(
        #     int
        # )
        size = np.int32((np.round(size) if np.allclose(np.round(size), size) else size))
        self._mshape = (size, size)
        return super().set_kbins(kmin, kmax, dk, nmodes)


class PowerSpectrumMultipolesCovariance(MultipoleFourierCovariance):
    """Covariance matrix of power spectrum multipoles in a given geometry.

    Attributes
    ----------
    geometry : geometry.Geometry
        Geometry of the survey. Can be a BoxGeometry or a SurveyGeometry object.
    """

    def __init__(self, geometry=None):
        MultipoleFourierCovariance.__init__(self)
        self.logger = logging.getLogger("PowerSpectrumCovariance")

        self.geometry = geometry

        self._pk = {}
        self._alpha = None

        self.pk_renorm = 1

    @property
    def alpha(self) -> Optional[float]:
        """The value of alpha. This is the alpha used in the Pk measurements.
           It can be different from the alpha used in the geometry object.

        Returns
        -------
        float
            The value of alpha.
        """
        if (
            self._alpha is None
            and self.geometry is not None
            and hasattr(self.geometry, "alpha")
        ):
            return self.geometry.alpha
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: Optional[float]):
        """Sets the value of alpha. This is the alpha used in the P(k) measurements.
           It can be different from the alpha used in the geometry object.

        Parameters
        ----------
        alpha : float
            The value of alpha.
        """
        self._alpha = alpha

    def compute_covariance(self, ells: tuple[int, ...] | list = (0, 2, 4)):
        """Compute the covariance matrix for the given geometry and power spectra.

        Parameters
        ----------
        ells : tuple[int, ...] | list[int], optional
            Multipoles of the power spectra to have the covariance calculated for.
        """

        self._ells = ells
        self._mshape = (self.kbins, self.kbins)

        if isinstance(self.geometry, geometry.BoxGeometry):
            return self._compute_covariance_box()

        if isinstance(self.geometry, geometry.SurveyGeometry):
            return self._compute_covariance_survey()

    def _compute_covariance_box(self):
        raise NotImplementedError

    def _compute_covariance_survey(self):
        raise NotImplementedError

    @property
    def shotnoise(self) -> Any:
        """Shotnoise of the sample in the same normalization as the power spectrum.

        Returns
        -------
        float
            Shotnoise value."""

        if isinstance(self.geometry, geometry.SurveyGeometry):
            assert self.alpha is not None, "Alpha is not set. Cannot compute shotnoise."
            return (
                self.pk_renorm
                * (1 + self.alpha)
                * self.geometry.I("12")
                / self.geometry.I("22")
            )
        elif isinstance(self.geometry, geometry.BoxGeometry):
            return self.pk_renorm * self.geometry.shotnoise

    def set_shotnoise(self, shotnoise: float):
        """Determines the relative normalization of the power spectrum by comparing
           the estimated FKP shotnoise with the given shotnoise value.

        Parameters
        ----------
        shotnoise : float
            shotnoise with same normalization as the power spectrum.
        """

        self.logger.info(f"Estimated shotnoise was {self.shotnoise}")
        self.logger.info(f"Forcing it to be {shotnoise}.")

        self.pk_renorm *= shotnoise / self.shotnoise
        self.logger.info(
            f"Setting pk_renorm to {self.pk_renorm} based on given shotnoise value."
        )
