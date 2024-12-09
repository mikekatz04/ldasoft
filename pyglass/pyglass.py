from dataclasses import dataclass

@dataclass
class GlassSettings:
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
    rj          = 1
    help        = 0
    calibration = 0
    verbose     = 0
    quiet       = 0
    simNoise    = 0
    confNoise   = 0
    burnin      = 1
    debug       = 0
    strainData  = 0
    hdf5Data    = 0
    psd         = 0
    orbit       = 0
    prior       = 0
    resume      = 0
    NMCMC       = 10000
    NBURN       = 10000
    threads     = 4  # omp_get_max_threads()
    runDir = "./"
    set_fmax_flag: int  = 0 #flag watching for if fmax is set by CLI
    fmin: float = 1e-4  # Hz
    qpad : int = 0
    # Simulated data building blocks */
    no_mbh: int = 0
    no_ucb: int = 0
    no_vgb: int = 0
    no_noise : int = 0

    runDir = "./"      #!<store `DIRECTORY` to serve as top level directory for output files.
    vbFile = "vb_file_test.dat"       #!<store `FILENAME` of list of known binaries `vb_mcmc`
    ucbGridFile = "ucb_file_test.dat"  #!<`[--ucb-grid=FILENAME]` frequency grid for multiband UCB analysis
    injFile = ["/data/mkatz/LISAanalysistools/ldasoft/ucb/etc/sources/precision/PrecisionSource_0.txt"]                 #!<`[--inj=FILENAME]`: list of injection files. Can support up to `NINJ=10` separate injections.
    noiseFile = "noise_test.dat"    #!<file containing reconstructed noise model for `gb_catalog` to compute SNRs against.
    cdfFile = "cdf_test.dat"      #!<store `FILENAME` of input chain file from Flags::update.
    gmmFile = "gmm_file_test.dat"      #!<store `FILENAME` of input gmm file from Flags::update.
    covFile = "cov_file_test.dat"      #!<store `FILENAME` of input covariance matrix file from Flags::updateCov.
    matchInfile1 = "match1_file_test.dat" #!<input waveform \f$A\f$ for computing match \f$(h_A|h_B)\f$
    matchInfile2 = "match2_file_test.dat" #!<input waveform \f$B\f$ for computing match \f$(h_A|h_B)\f$
    pdfFile = "pdf_file_test.dat"     #!<store `FILENAME` of input priors for Flags:knownSource.
    psdFile = "psd_file_test.dat"      #!<store `FILENAME` of input psd file from Flags::psd.
    catalogFile = "cat_file_test.dat"  #!<store `FILENAME` containing previously identified detections from Flags::catalog for cleaning padding regions

    format      : str = "sangria"
    dataDir     : str = "./data_store"
    N           : int = 512
    Nchannel    : int = 2
    cseed       : int =150914
    nseed : int =151226
    iseed: int =151012

    NC : int = 12
    
    T: float = 36000000.0
    t0: float = 0.0
    
    Nwave : int = 10
    downsample : int = 2
    fileName : str = "temp_test_fileName.dat"
  
