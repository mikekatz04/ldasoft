import numpy as np
cimport numpy as np
from libc cimport bool
from libc.stdint cimport uintptr_t
from libc.string cimport strcpy, strlen
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free


cdef extern from "mixglass.h":
    cdef struct Translator:
        int verbose;    #!<`[--verbose; default=FALSE]`: increases file output (all chains, less downsampling, etc.)
        int quiet;      #!<`[--quiet; default=FALSE]`: decreases file output (less diagnostic data, only what is needed for post processing/recovery)
        int NMCMC;      #!<`[--steps=INT; default=100000]`: number of MCMC samples
        int NBURN;      #!<number of burn in samples, equals Flags::NMCMC.
        int NINJ;       #!<`[--inj=FILENAME]`: number of injections = number of `--inj` instances in command line
        int NVB;        #!<number of known binaries for `vb_mcmc`
        int DMAX;       #!<`[--sources=INT; default=10]`: max number of sources
        int simNoise;   #!<`[--sim-noise; default=FALSE]`: simulate random noise realization and add to data
        int fixSky;     #!<`[--fix-sky; default=FALSE]`: hold sky location fixed to injection parameters.  Set to `TRUE` if Flags::knownSource=`TRUE`.
        int fixFdot;
        int fixFreq;    #!<`[--fix-freq; default=FALSE]`: hold GW frequency fixed to injection parameters
        int galaxyPrior;#!<`[--galaxy-prior; default=FALSE]`: use model of galaxy for sky location prior
        int snrPrior;   #!<`[--snr-prior; default=FALSE]`: use SNR prior for amplitude
        int emPrior;    #!<`[--em-prior=FILENAME]`: use input data file with EM-derived parameters for priors.
        int knownSource;#!<`[--known-source; default=FALSE]`: injection is known binary, will need polarization and phase to be internally generated. Sets Flags::fixSky = `TRUE`.
        int detached;   #!<`[--detached; default=FALSE]`: assume binary is detached, fdot prior becomes \f$U[\dot{f}(\mathcal{M}_c=0.15),\dot{f}(\mathcal{M}_c=1.00)]\f$
        int strainData; #!<`[--data=FILENAME; default=FALSE]`: read data from ASCII file instead of simulate internally.
        int hdf5Data;   #!<'[--hdf5Data=FILENAME; default=FALSE]`: read data from LDC HDF5 file (compatible w/ Sangria dataset).
        int orbit;      #!<`[--orbit=FILENAME; default=FALSE]`: use numerical spacecraft ephemerides supplied in `FILENAME`. `--orbit` argument sets flag to `TRUE`.
        int prior;      #!<`[--prior; default=FALSE]`: set log-likelihood to constant for testing detailed balance.
        int debug;      #!<`[--debug; default=FALSE]`: coarser settings for proposals and verbose output for debugging
        int cheat;      #!<start sampler at injection values
        int burnin;     #!<`[--no-burnin; default=TRUE]`: chain is in the burn in phase
        int maximize;   #!<maximize over extrinsic parameter during burn in phase.
        int update;     #!<`[--update=FILENAME; default=FALSE]`: use Gaussian Mixture Model approximation to previous posterior as current prior.
        int updateCov;  #!<`[--update-cov=FILENAME; default=FALSE]`: updating fit from covariance matrix files built from chain samples, passed as `FILENAME`, used in draw_from_cov().
        int match;      #!<[--match=FLOAT; default=0.8]`: match threshold for chain sample clustering in post processing.
        int rj;         #!<--no-rj; default=TRUE]`: flag for determining if trans dimensional MCMC moves (RJMCMC) are enabled.
        int calibration;#!<`[--calibration; default=FALSE]`: flag for determining if model is marginalizing over calibration  uncertainty.
        int confNoise;  #!<`[--conf-noise; default=FALSE]`: include model of confusion noise in \f$S_n(f)\f$, either for simulating noise or as starting value for parameterized noise model.
        int resume;     #!<`[--resume; default=FALSE]`: restart sampler from run state saved during checkpointing. Starts from scratch if no checkpointing files are found.
        int catalog;    #!<`[--catalog=FILENAME; default=FALSE]`: use list of previously detected sources supplied in `FILENAME` to clean bandwidth padding (`gb_mcmc`) or for building family tree (`gb_catalog`).
        int grid;       #!<`[--ucb-grid=FILENAME; default=FALSE]`: flag indicating if a gridfile was supplied
        int threads;    #!<number of openMP threads for parallel tempering
        int psd;        #!<`[--psd=FILENAME; default=FALSE]`: use PSD input as ASCII file from command line
        int help;       #!<`[--help]`: print command line usage and exit
        int qpad;
        double fmin; 

        char runDir[1024];       #!<store `DIRECTORY` to serve as top level directory for output files.
        char vbFile[1024];       #!<store `FILENAME` of list of known binaries `vb_mcmc`
        char ucbGridFile[1024];  #!<`[--ucb-grid=FILENAME]` frequency grid for multiband UCB analysis
        char **injFile;                   #!<`[--inj=FILENAME]`: list of injection files. Can support up to `NINJ=10` separate injections.
        char noiseFile[1024];    #!<file containing reconstructed noise model for `gb_catalog` to compute SNRs against.
        char cdfFile[1024];      #!<store `FILENAME` of input chain file from Flags::update.
        char gmmFile[1024];      #!<store `FILENAME` of input gmm file from Flags::update.
        char covFile[1024];      #!<store `FILENAME` of input covariance matrix file from Flags::updateCov.
        char matchInfile1[1024]; #!<input waveform \f$A\f$ for computing match \f$(h_A|h_B)\f$
        char matchInfile2[1024]; #!<input waveform \f$B\f$ for computing match \f$(h_A|h_B)\f$
        char pdfFile[1024];      #!<store `FILENAME` of input priors for Flags:knownSource.
        char psdFile[1024];      #!<store `FILENAME` of input psd file from Flags::psd.
        char catalogFile[1024];  #!<store `FILENAME` containing previously identified detections from Flags::catalog for cleaning padding regions
    
        int no_mbh;
        int no_ucb;
        int no_ucb_hi;
        int no_vgb;
        int no_noise;

        int N;        #!<number of frequency bins
        int Nchannel; #!<number of data channels
        
        long cseed; #!<seed for MCMC set by `[--chainseed=INT; default=150914]`
        long nseed; #!<seed for noise realization set by `[--noiseseed=INT; default=151226]`
        long iseed; #!<seed for injection parameters set by `[--injseed=INT; default=151012]`
        double T; #!<observation time
        double t0;   #!<start times of segments
        int Nwave; #!<Number of samples for computing posterior reconstructions
        int downsample; #!<Downsample factor for getting the desired number of samples
        char fileName[1024]; #!<place holder for filnames
        char format[16];
        char dataDir[1024]; 

        int NC;

cdef extern from "mixglass.h":
    void glass_wrapper(Translator *translator);
    void setup_ucb_global_fit_main(Translator *translator, int procID, int procID_min, int procID_max, int nUCB);
    void clear_ucb_global_fit_main(Translator *translator, int procID, int procID_min, int procID_max);
    void setup_ucb_global_fit_child(Translator *translator, int procID, int procID_min, int procID_max, int nUCB);
    void clear_ucb_global_fit_child(Translator *translator, int procID, int procID_min, int procID_max);
    void run_glass_ucb_step(int step, Translator *translator, int procID, int procID_min, int procID_max);
    void get_current_glass_params(Translator *translator, double *params, int *nleaves, double *logl, double *logp, double *betas, int array_length, int ucb_index);
    void set_current_glass_params(Translator *translator, double *params, int *nleaves, double *logl, double *logp, double *betas, int array_length, int ucb_index);
    void get_current_cold_chain_glass_residual(Translator *translator, double *data_arr, int Nchannel, int N);
    void get_psd_in_glass(Translator *translator, double *noise_arr, int Nchannel, int N);
    void set_psd_in_glass(Translator *translator, double *noise_arr, int Nchannel, int N);
    void get_main_tdi_data_in_glass(Translator *translator, double *data_arr, int Nchannel, int N);
    void set_main_tdi_data_in_glass(Translator *translator, double *data_arr, int Nchannel, int N);
    void mpi_process_runner(int procID, Translator *translator, int procID_min);
    void mpi_process_send_out(int procID, Translator *translator, int procID_min, int next_free_process, int ucb_index);
    int get_frequency_domain_data_length(Translator *translator);
    double get_frequency_domain_data_df(Translator *translator);
    
cdef void setup_translator(Translator *translator, glass_settings):

    translator.verbose = glass_settings.verbose;    #!<`[--verbose; default=FALSE]`: increases file output (all chains, less downsampling, etc.)
    translator.quiet = glass_settings.quiet;      #!<`[--quiet; default=FALSE]`: decreases file output (less diagnostic data, only what is needed for post processing/recovery)
    translator.NMCMC = glass_settings.NMCMC;      #!<`[--steps=INT; default=100000]`: number of MCMC samples
    translator.NBURN = glass_settings.NBURN;      #!<number of burn in samples, equals Flags::NMCMC.
    translator.NINJ = glass_settings.NINJ;       #!<`[--inj=FILENAME]`: number of injections = number of `--inj` instances in command line
    # translator.NVB = glass_settings.NVB;        #!<number of known binaries for `vb_mcmc`
    translator.DMAX = glass_settings.DMAX;       #!<`[--sources=INT; default=10]`: max number of sources
    translator.simNoise = glass_settings.simNoise;   #!<`[--sim-noise; default=FALSE]`: simulate random noise realization and add to data
    translator.fixSky = glass_settings.fixSky;     #!<`[--fix-sky; default=FALSE]`: hold sky location fixed to injection parameters.  Set to `TRUE` if Flags::knownSource=`TRUE`.
    translator.fixFdot = glass_settings.fixFdot;
    translator.fixFreq = glass_settings.fixFreq;    #!<`[--fix-freq; default=FALSE]`: hold GW frequency fixed to injection parameters
    translator.galaxyPrior = glass_settings.galaxyPrior;#!<`[--galaxy-prior; default=FALSE]`: use model of galaxy for sky location prior
    translator.snrPrior = glass_settings.snrPrior;   #!<`[--snr-prior; default=FALSE]`: use SNR prior for amplitude
    translator.emPrior = glass_settings.emPrior;    #!<`[--em-prior=FILENAME]`: use input data file with EM-derived parameters for priors.
    translator.knownSource = glass_settings.knownSource;#!<`[--known-source; default=FALSE]`: injection is known binary, will need polarization and phase to be internally generated. Sets Flags::fixSky = `TRUE`.
    translator.detached = glass_settings.detached;   #!<`[--detached; default=FALSE]`: assume binary is detached, fdot prior becomes \f$U[\dot{f}(\mathcal{M}_c=0.15),\dot{f}(\mathcal{M}_c=1.00)]\f$
    translator.strainData = glass_settings.strainData; #!<`[--data=FILENAME; default=FALSE]`: read data from ASCII file instead of simulate internally.
    translator.hdf5Data = glass_settings.hdf5Data;   #!<'[--hdf5Data=FILENAME; default=FALSE]`: read data from LDC HDF5 file (compatible w/ Sangria dataset).
    translator.orbit = glass_settings.orbit;      #!<`[--orbit=FILENAME; default=FALSE]`: use numerical spacecraft ephemerides supplied in `FILENAME`. `--orbit` argument sets flag to `TRUE`.
    translator.prior = glass_settings.prior;      #!<`[--prior; default=FALSE]`: set log-likelihood to constant for testing detailed balance.
    translator.debug = glass_settings.debug;      #!<`[--debug; default=FALSE]`: coarser settings for proposals and verbose output for debugging
    translator.cheat = glass_settings.cheat;      #!<start sampler at injection values
    translator.burnin = glass_settings.burnin;     #!<`[--no-burnin; default=TRUE]`: chain is in the burn in phase
    # translator.maximize = glass_settings.maximize;   #!<maximize over extrinsic parameter during burn in phase.
    translator.update = glass_settings.update;     #!<`[--update=FILENAME; default=FALSE]`: use Gaussian Mixture Model approximation to previous posterior as current prior.
    translator.updateCov = glass_settings.updateCov;  #!<`[--update-cov=FILENAME; default=FALSE]`: updating fit from covariance matrix files built from chain samples, passed as `FILENAME`, used in draw_from_cov().
    translator.match = glass_settings.match;      #!<[--match=FLOAT; default=0.8]`: match threshold for chain sample clustering in post processing.
    translator.rj = glass_settings.rj;         #!<--no-rj; default=TRUE]`: flag for determining if trans dimensional MCMC moves (RJMCMC) are enabled.
    translator.calibration = glass_settings.calibration;#!<`[--calibration; default=FALSE]`: flag for determining if model is marginalizing over calibration  uncertainty.
    translator.confNoise = glass_settings.confNoise;  #!<`[--conf-noise; default=FALSE]`: include model of confusion noise in \f$S_n(f)\f$, either for simulating noise or as starting value for parameterized noise model.
    translator.resume = glass_settings.resume;     #!<`[--resume; default=FALSE]`: restart sampler from run state saved during checkpointing. Starts from scratch if no checkpointing files are found.
    translator.catalog = glass_settings.catalog;    #!<`[--catalog=FILENAME; default=FALSE]`: use list of previously detected sources supplied in `FILENAME` to clean bandwidth padding (`gb_mcmc`) or for building family tree (`gb_catalog`).
    translator.grid = glass_settings.grid;       #!<`[--ucb-grid=FILENAME; default=FALSE]`: flag indicating if a gridfile was supplied
    translator.threads = glass_settings.threads;    #!<number of openMP threads for parallel tempering
    translator.psd = glass_settings.psd;        #!<`[--psd=FILENAME; default=FALSE]`: use PSD input as ASCII file from command line
    translator.help = glass_settings.help;       #!<`[--help]`: print command line usage and exit

    translator.no_mbh = glass_settings.no_mbh;
    translator.no_ucb = glass_settings.no_ucb;
    translator.no_ucb_hi = glass_settings.no_ucb_hi;
    translator.no_vgb = glass_settings.no_vgb;
    translator.no_noise = glass_settings.no_noise;

    strcpy(translator.dataDir, glass_settings.dataDir.encode('utf-8'))

    translator.N = glass_settings.N
    translator.Nchannel = glass_settings.Nchannel
    translator.cseed = glass_settings.cseed
    translator.nseed = glass_settings.nseed
    translator.iseed = glass_settings.iseed
    translator.T = glass_settings.T
    translator.t0 = glass_settings.iseed
    translator.Nwave = glass_settings.Nwave
    translator.downsample = glass_settings.downsample
    translator.qpad = glass_settings.qpad
    translator.fmin = glass_settings.fmin

    strcpy(translator.fileName, glass_settings.fileName.encode('utf-8'))
    strcpy(translator.format, glass_settings.format.encode('utf-8'))
    
    strcpy(translator.runDir, glass_settings.runDir.encode('utf-8'))
    strcpy(translator.vbFile, glass_settings.vbFile.encode('utf-8'))
    strcpy(translator.ucbGridFile, glass_settings.ucbGridFile.encode('utf-8'))
    strcpy(translator.noiseFile, glass_settings.noiseFile.encode('utf-8'))
    strcpy(translator.cdfFile, glass_settings.cdfFile.encode('utf-8'))
    strcpy(translator.gmmFile, glass_settings.gmmFile.encode('utf-8'))
    strcpy(translator.covFile, glass_settings.covFile.encode('utf-8'))
    strcpy(translator.matchInfile1, glass_settings.matchInfile1.encode('utf-8'))
    strcpy(translator.matchInfile2, glass_settings.matchInfile2.encode('utf-8'))
    strcpy(translator.pdfFile, glass_settings.pdfFile.encode('utf-8'))
    strcpy(translator.psdFile, glass_settings.psdFile.encode('utf-8'))
    strcpy(translator.catalogFile, glass_settings.catalogFile.encode('utf-8'))
    
    translator.NC = glass_settings.NC

    assert len(glass_settings.injFile) == translator.NINJ
    return 

def run_glass_ucb_mcmc(glass_settings):
    
    cdef Translator translator;
    setup_translator(&translator, glass_settings)
    
    assert len(glass_settings.injFile) == translator.NINJ

    translator.injFile = <char**>malloc(translator.DMAX *sizeof(char *));
    for n in range(translator.DMAX):
        translator.injFile[n] = <char*>malloc(1024*sizeof(char));

    for n in range(translator.NINJ):
        strcpy(translator.injFile[n], glass_settings.injFile[n].encode('utf-8'))
    
    glass_wrapper(&translator)

    printf(translator.injFile[0])
    printf(translator.catalogFile)

    for n in range(translator.DMAX):
        free(translator.injFile[n]);
    free(translator.injFile);

    
cdef class GlassGlobalFitTranslate:
    cdef Translator *translator;
    cdef int procID, procID_min, procID_max;
    cdef int dmax;
    cdef int nparams;
    cdef int NC;
    cdef int Nchannel;
    cdef int N;
    cdef int is_main;
    cdef int nUCB;
    cdef double df;

    def __cinit__(self, procID, procID_min, procID_max, nUCB):
        self.translator = <Translator*>malloc(sizeof(Translator));
        self.procID, self.procID_min, self.procID_max = procID, procID_min, procID_max
        self.nparams = 8;
        self.is_main = int(procID == procID_min)
        self.nUCB = nUCB
        
    def setup_ucb_global_fit_main(self, glass_settings):

        self.dmax = glass_settings.DMAX
        self.NC = glass_settings.NC
        self.Nchannel = glass_settings.Nchannel
        
        setup_translator(self.translator, glass_settings)
        
        self.translator.injFile = <char**>malloc(self.translator.DMAX *sizeof(char *));
        for n in range(self.translator.DMAX):
            self.translator.injFile[n] = <char*>malloc(1024*sizeof(char));

        for n in range(self.translator.NINJ):
            strcpy(self.translator.injFile[n], glass_settings.injFile[n].encode('utf-8'))
        
        setup_ucb_global_fit_main(self.translator, self.procID, self.procID_min, self.procID_max, self.nUCB);

        for n in range(self.translator.DMAX):
            free(self.translator.injFile[n]);
        free(self.translator.injFile);

        self.N = get_frequency_domain_data_length(self.translator)
        self.df = get_frequency_domain_data_df(self.translator)

    @property
    def df(self):
        return self.df
    
    @property
    def N(self):
        return self.N
        
    def setup_ucb_global_fit_child(self, glass_settings):

        self.dmax = glass_settings.DMAX
        self.NC = glass_settings.NC
        self.Nchannel = glass_settings.Nchannel
        
        setup_translator(self.translator, glass_settings)
        
        self.translator.injFile = <char**>malloc(self.translator.DMAX *sizeof(char *));
        for n in range(self.translator.DMAX):
            self.translator.injFile[n] = <char*>malloc(1024*sizeof(char));

        for n in range(self.translator.NINJ):
            strcpy(self.translator.injFile[n], glass_settings.injFile[n].encode('utf-8'))
        
        setup_ucb_global_fit_child(self.translator, self.procID, self.procID_min, self.procID_max, self.nUCB);

        for n in range(self.translator.DMAX):
            free(self.translator.injFile[n]);
        free(self.translator.injFile);

        self.N = get_frequency_domain_data_length(self.translator)
        self.df = get_frequency_domain_data_df(self.translator)

    def run_glass_ucb_step(self, step : int):
        run_glass_ucb_step(step, self.translator, self.procID, self.procID_min, self.procID_max);

    def get_main_tdi_data_in_glass(self):
        # glass uses double storage for complex
        # we will convert in python

        cdef np.ndarray[np.float64_t, ndim=1] tdi_channels_out = np.zeros((self.Nchannel * self.N * 2,), dtype=np.float64)
        
        get_main_tdi_data_in_glass(self.translator, &tdi_channels_out[0], self.Nchannel, self.N);

        output = np.zeros((self.Nchannel, self.N), dtype=complex)
        output[:] = (
            tdi_channels_out.reshape(self.Nchannel, 2 * self.N)[:, 0::2]
            + 1j * tdi_channels_out.reshape(self.Nchannel, 2 * self.N)[:, 1::2]
        )
        return output

    def set_main_tdi_data_in_glass(self,
            tdi,  
        ):

        assert tdi.shape == (self.Nchannel, self.N)
        cdef np.ndarray[np.float64_t, ndim=1] tdi_in = np.zeros((self.Nchannel * self.N * 2))
        
        tdi_in[0::2] = tdi.real.flatten()
        tdi_in[1::2] = tdi.imag.flatten()
        
        set_main_tdi_data_in_glass(self.translator, &tdi_in[0], self.Nchannel, self.N);

    def get_current_cold_chain_glass_residual(self):
        # glass uses double storage for complex
        # we will convert in python

        cdef np.ndarray[np.float64_t, ndim=1] tdi_channels_out = np.zeros((self.Nchannel * self.N * 2,), dtype=np.float64)
        
        get_current_cold_chain_glass_residual(self.translator, &tdi_channels_out[0], self.Nchannel, self.N);

        output = np.zeros((self.Nchannel, self.N), dtype=complex)
        output[:] = (
            tdi_channels_out.reshape(self.Nchannel, 2 * self.N)[:, 0::2]
            + 1j * tdi_channels_out.reshape(self.Nchannel, 2 * self.N)[:, 1::2]
        )
        return output

    def set_psd_in_glass(self,
            psd,  
        ):

        if self.Nchannel == 2:
            shape = (self.Nchannel, self.N)
        elif self.Nchannel == 3:
            shape = (self.Nchannel, self.Nchannel, self.N)

        if not psd.shape == shape:
            raise ValueError(f"psd must have shape (Nchannel, N) if 2 channel or (Nchannel, Nchannel, N) if 3 channel. Current shape is {psd.shape} with {self.Nchannel}.")

        cdef np.ndarray[np.float64_t, ndim=1] psd_in = np.zeros(np.prod(shape), dtype=np.float64)
        
        psd_in[:] = psd.flatten()
        
        set_psd_in_glass(self.translator, &psd_in[0], self.Nchannel, self.N);

    def get_psd_in_glass(self):
        
        if self.Nchannel == 2:
            shape = (self.Nchannel, self.N)
        elif self.Nchannel == 3:
            shape = (self.Nchannel, self.Nchannel, self.N)

        cdef np.ndarray[np.float64_t, ndim=1] psd_in = np.zeros(np.prod(shape), dtype=np.float64)
        
        get_psd_in_glass(self.translator, &psd_in[0], self.Nchannel, self.N);
        return psd_in.reshape(shape)

    def set_current_glass_params(self,
            ucb_index,
            np.ndarray[np.float64_t, ndim=1] params,
            np.ndarray[np.int32_t, ndim=1] nleaves,
            np.ndarray[np.float64_t, ndim=1] logl,
            np.ndarray[np.float64_t, ndim=1] logp,
            np.ndarray[np.float64_t, ndim=1] betas
        ):

        assert ucb_index >= 0
        assert ucb_index < self.nUCB
        
        assert len(nleaves) == len(logl) == len(logp) == len(betas) == self.NC
        assert len(params) == self.NC * self.dmax * self.nparams

        set_current_glass_params(self.translator, &params[0], <int*>&nleaves[0], &logl[0], &logp[0], &betas[0], self.NC * self.dmax, ucb_index);

    def get_current_glass_params(self, ucb_index):
        cdef np.ndarray[np.float64_t, ndim=1] params = np.zeros((self.NC *self.dmax * self.nparams,), dtype=np.float64)
        cdef np.ndarray[np.int32_t, ndim=1] nleaves = np.zeros((self.NC,), dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1] logl = np.zeros((self.NC,), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] logp = np.zeros((self.NC,), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] betas = np.zeros((self.NC,), dtype=np.float64)
        
        assert ucb_index >= 0
        assert ucb_index < self.nUCB

        get_current_glass_params(self.translator, &params[0], <int*>&nleaves[0], &logl[0], &logp[0], &betas[0], self.NC * self.dmax, ucb_index);

        return (
            params.reshape(self.NC, self.dmax, self.nparams),
            nleaves,
            logl,
            logp,
            betas
        )

    def __dealloc__(self):
        if self.translator: 
            if self.is_main == 1:  # bool not working for some reason.
                clear_ucb_global_fit_main(self.translator, self.procID, self.procID_min, self.procID_max);
            
            else:
                clear_ucb_global_fit_child(self.translator, self.procID, self.procID_min, self.procID_max);
            
            print("freeing translator")
            free(self.translator)
    def mpi_process_runner(self):
        mpi_process_runner(self.procID, self.translator, self.procID_min);

    def mpi_process_send_out(self, procID_target, next_free_process, ucb_index):
        mpi_process_send_out(self.procID, self.translator, procID_target, next_free_process, ucb_index);

    def __reduce__(self):
        return (rebuild, (self.procID, self.procID_min, self.procID_max,))

def rebuild(procID, procID_min, procID_max):
    c = GlassGlobalFitTranslate(procID, procID_min, procID_max)
    return c

