# from future.utils import iteritems
import os
from os.path import join as pjoin
from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Distutils import build_ext

# from Cython.Distutils import build_ext
import numpy
import shutil


# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# try:
#     import mpi4py
#     mpi_include_dir = mpi4py.get_include() + "/mpi4py"

# except ModuleNotFoundError:
#     mpi_include_dir = None
#     print("No MPI/mpi4py.")

mpi_compile_args = os.popen("mpicc --showme:compile").read().strip().split(' ')
mpi_link_args    = os.popen("mpicc --showme:link").read().strip().split(' ')

glass_ext = Extension(
    "pyglass.translateglass",
    sources=[
        "utils/src/glass_data.c",
        "utils/src/glass_math.c",
        "utils/src/glass_lisa.c",
        "utils/src/glass_gmm.c",
        "ucb/src/gb_mcmc_frequency_spacing.c",
        "ucb/src/gb_mcmc_frequency_spacing_2.c",
        "ucb/src/glass_ucb_data.c",
        "ucb/src/glass_ucb_fstatistic.c",
        "ucb/src/glass_ucb_io.c",
        "ucb/src/glass_ucb_model.c",
        "ucb/src/glass_ucb_prior.c",
        "ucb/src/glass_ucb_proposal.c",
        "ucb/src/glass_ucb_residual.c",
        "ucb/src/glass_ucb_sampler.c",
        "ucb/src/glass_ucb_catalog.c",
        "ucb/src/glass_ucb_waveform.c",
        "noise/src/glass_noise_model.c",
        "noise/src/glass_noise_io.c",
        "noise/src/glass_noise_sampler.c",
        "globalfit/src/glass_mbh_wrapper.c",
        "globalfit/src/glass_noise_wrapper.c",
        "globalfit/src/glass_vgb_wrapper.c",
        "apps/src/ucb_mcmc_base.c",
        "globalfit/src/glass_ucb_wrapper.c",
        "LISA-Massive-Black-Hole/PTMCMC.c",
        "LISA-Massive-Black-Hole/IMRPhenomD.c",
        "LISA-Massive-Black-Hole/Utilities.c",
        "LISA-Massive-Black-Hole/SpecFit.c",
        # "LISA-Massive-Black-Hole/segmentSangria.c",
        "LISA-Massive-Black-Hole/Response.c",
        "LISA-Massive-Black-Hole/IMRPhenomD_internals.c",
        "LISA-Massive-Black-Hole/Utils.c",
        # "LISA-Massive-Black-Hole/unique.c",
        # "LISA-Massive-Black-Hole/SpecAverage.c",
        # "LISA-Massive-Black-Hole/search.c",
        "globalfit/src/globalfit.c",
        "utils/src/mixglass.c",
        "utils/src/glass_translator.pyx",
    ],
    libraries=["gsl", "gslcblas", "pthread", "hdf5", "gomp"],
    include_dirs=[numpy_include, "ucb/src", "utils/src", "apps/src", "globalfit/src", "noise/src", "LISA-Massive-Black-Hole"],
    language="c",
    extra_compile_args=["-c", "-fopenmp", "-w"] + mpi_compile_args,
    extra_link_args=mpi_link_args
)

extensions = [glass_ext]
with open("README.md", "r") as fh:
    long_description = fh.read()

# setup version file
with open("README.md", "r") as fh:
    lines = fh.readlines()

setup(
    name="ldasoft",
    author="Tyson Littenberg",
    author_email="tyson.b.littenberg@nasa.gov",
    ext_modules=extensions,
    # Inject our custom trigger
    packages=["pyglass"],
    # Since the package has c code, the egg cannot be zipped
    zip_safe=False,
    url="https://github.com/tliteenberg/ldasoft",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
        "Programming Language :: C",
        "Programming Language :: Cython",
    ],
    python_requires=">=3.6",
)
