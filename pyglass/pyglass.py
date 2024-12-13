from __future__ import annotations
from dataclasses import dataclass
from typing import List


class GlassSettings:
    verbose : int = 1
    NINJ        : int =1
    fixSky      : int =0
    fixFreq     : int =0
    fixFdot     : int =0
    galaxyPrior : int =0
    snrPrior    : int =1
    emPrior     : int =0
    cheat       : int =0
    detached    : int =0
    knownSource : int =0
    catalog     : int =0
    grid        : int =0
    update      : int =0
    updateCov   : int =0
    match       : int =0
    DMAX        : int =10


    #Set defaults
    rj          : int = 1
    help        : int = 0
    calibration : int = 0
    quiet       : int = 0
    simNoise    : int = 0
    confNoise   : int = 0
    burnin      : int = 1
    debug       : int = 0
    strainData  : int = 0
    hdf5Data    : int = 0
    psd         : int = 0
    orbit       : int = 0
    prior       : int = 0
    resume      : int = 0
    NMCMC       : int = 10000
    NBURN       : int = 10000
    threads     : int = 1  # omp_get_max_threads()
    runDir : str = "./"
    # set_fmax_flag: int  = 0 #flag watching for if fmax is set by CLI
    fmin: float = 1e-4  # Hz
    qpad : int = 0
    # Simulated data building blocks */
    no_mbh: int = 1
    no_ucb: int = 0
    no_ucb_hi: int = 0
    no_vgb: int = 1
    no_noise : int = 1

    runDir : str = "./"      #!<store `DIRECTORY` to serve as top level directory for output files.
    vbFile : str = "vb_file_test.dat"       #!<store `FILENAME` of list of known binaries `vb_mcmc`
    ucbGridFile : str = "ucb_file_test.dat"  #!<`[--ucb-grid=FILENAME]` frequency grid for multiband UCB analysis
    injFile : list = ["/data/mkatz/LISAanalysistools/ldasoft/ucb/etc/sources/precision/PrecisionSource_0.txt"]                 #!<`[--inj=FILENAME]`: list of injection files. Can support up to `NINJ=10` separate injections.
    noiseFile : str = "noise_test.dat"    #!<file containing reconstructed noise model for `gb_catalog` to compute SNRs against.
    cdfFile : str = "cdf_test.dat"      #!<store `FILENAME` of input chain file from Flags::update.
    gmmFile : str = "gmm_file_test.dat"      #!<store `FILENAME` of input gmm file from Flags::update.
    covFile : str = "cov_file_test.dat"      #!<store `FILENAME` of input covariance matrix file from Flags::updateCov.
    matchInfile1 : str = "match1_file_test.dat" #!<input waveform \f$A\f$ for computing match \f$(h_A|h_B)\f$
    matchInfile2 : str = "match2_file_test.dat" #!<input waveform \f$B\f$ for computing match \f$(h_A|h_B)\f$
    pdfFile : str = "pdf_file_test.dat"     #!<store `FILENAME` of input priors for Flags:knownSource.
    psdFile : str = "psd_file_test.dat"      #!<store `FILENAME` of input psd file from Flags::psd.
    catalogFile : str = "cat_file_test.dat"  #!<store `FILENAME` containing previously identified detections from Flags::catalog for cleaning padding regions
    h5_data : str = ""

    format      : str = "sangria"
    dataDir     : str = "./data_store"
    N           : int = 512
    Nchannel    : int = 2
    cseed       : int =150914
    nseed : int =151226
    iseed: int =151012

    NC : int = 12
    
    T: float = 31000000.0
    t0: float = 0.0
    
    Nwave : int = 10
    downsample : int = 2
    fileName : str = "../LDC2_sangria_training_v2.h5"
  
    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(f"GlassSettings class does not have the argument {key}.")

            setattr(self, key, value)
