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
        "apps/src/ucb_mcmc_base.c",
        "utils/src/mixglass.c",
        "utils/src/glass_translator.pyx",
    ],
    libraries=["gsl", "gslcblas", "pthread", "hdf5", "gomp"],
    include_dirs=[numpy_include, "ucb/src", "utils/src", "apps/src"],
    language="c",
    extra_compile_args=["-c", "-fopenmp"],
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
