#ifndef mix_glass_h
#define mix_glass_h

#include <glass_utils.h>
#include <glass_ucb.h>
#include "glass_data.h"
#include "glass_ucb.h"
#include "glass_lisa.h"

struct Translator{
    int NC; /// Number of chains
    
    int verbose;    //!<`[--verbose; default=FALSE]`: increases file output (all chains, less downsampling, etc.)
    int quiet;      //!<`[--quiet; default=FALSE]`: decreases file output (less diagnostic data, only what is needed for post processing/recovery)
    int NMCMC;      //!<`[--steps=INT; default=100000]`: number of MCMC samples
    int NBURN;      //!<number of burn in samples, equals Flags::NMCMC.
    int NINJ;       //!<`[--inj=FILENAME]`: number of injections = number of `--inj` instances in command line
    int NVB;        //!<number of known binaries for `vb_mcmc`
    int DMAX;       //!<`[--sources=INT; default=10]`: max number of sources
    int simNoise;   //!<`[--sim-noise; default=FALSE]`: simulate random noise realization and add to data
    int fixSky;     //!<`[--fix-sky; default=FALSE]`: hold sky location fixed to injection parameters.  Set to `TRUE` if Flags::knownSource=`TRUE`.
    int fixFdot;
    int fixFreq;    //!<`[--fix-freq; default=FALSE]`: hold GW frequency fixed to injection parameters
    int galaxyPrior;//!<`[--galaxy-prior; default=FALSE]`: use model of galaxy for sky location prior
    int snrPrior;   //!<`[--snr-prior; default=FALSE]`: use SNR prior for amplitude
    int emPrior;    //!<`[--em-prior=FILENAME]`: use input data file with EM-derived parameters for priors.
    int knownSource;//!<`[--known-source; default=FALSE]`: injection is known binary, will need polarization and phase to be internally generated. Sets Flags::fixSky = `TRUE`.
    int detached;   //!<`[--detached; default=FALSE]`: assume binary is detached, fdot prior becomes \f$U[\dot{f}(\mathcal{M}_c=0.15),\dot{f}(\mathcal{M}_c=1.00)]\f$
    int strainData; //!<`[--data=FILENAME; default=FALSE]`: read data from ASCII file instead of simulate internally.
    int hdf5Data;   //!<'[--hdf5Data=FILENAME; default=FALSE]`: read data from LDC HDF5 file (compatible w/ Sangria dataset).
    int orbit;      //!<`[--orbit=FILENAME; default=FALSE]`: use numerical spacecraft ephemerides supplied in `FILENAME`. `--orbit` argument sets flag to `TRUE`.
    int prior;      //!<`[--prior; default=FALSE]`: set log-likelihood to constant for testing detailed balance.
    int debug;      //!<`[--debug; default=FALSE]`: coarser settings for proposals and verbose output for debugging
    int cheat;      //!<start sampler at injection values
    int burnin;     //!<`[--no-burnin; default=TRUE]`: chain is in the burn in phase
    int maximize;   //!<maximize over extrinsic parameter during burn in phase.
    int update;     //!<`[--update=FILENAME; default=FALSE]`: use Gaussian Mixture Model approximation to previous posterior as current prior.
    int updateCov;  //!<`[--update-cov=FILENAME; default=FALSE]`: updating fit from covariance matrix files built from chain samples, passed as `FILENAME`, used in draw_from_cov().
    int match;      //!<[--match=FLOAT; default=0.8]`: match threshold for chain sample clustering in post processing.
    int rj;         //!<--no-rj; default=TRUE]`: flag for determining if trans dimensional MCMC moves (RJMCMC) are enabled.
    int calibration;//!<`[--calibration; default=FALSE]`: flag for determining if model is marginalizing over calibration  uncertainty.
    int confNoise;  //!<`[--conf-noise; default=FALSE]`: include model of confusion noise in \f$S_n(f)\f$, either for simulating noise or as starting value for parameterized noise model.
    int resume;     //!<`[--resume; default=FALSE]`: restart sampler from run state saved during checkpointing. Starts from scratch if no checkpointing files are found.
    int catalog;    //!<`[--catalog=FILENAME; default=FALSE]`: use list of previously detected sources supplied in `FILENAME` to clean bandwidth padding (`gb_mcmc`) or for building family tree (`gb_catalog`).
    int grid;       //!<`[--ucb-grid=FILENAME; default=FALSE]`: flag indicating if a gridfile was supplied
    int threads;    //!<number of openMP threads for parallel tempering
    int psd;        //!<`[--psd=FILENAME; default=FALSE]`: use PSD input as ASCII file from command line
    int help;       //!<`[--help]`: print command line usage and exit
    
    char runDir[MAXSTRINGSIZE];       //!<store `DIRECTORY` to serve as top level directory for output files.
    char vbFile[MAXSTRINGSIZE];       //!<store `FILENAME` of list of known binaries `vb_mcmc`
    char ucbGridFile[MAXSTRINGSIZE];  //!<`[--ucb-grid=FILENAME]` frequency grid for multiband UCB analysis
    char **injFile;                   //!<`[--inj=FILENAME]`: list of injection files. Can support up to `NINJ=10` separate injections.
    char noiseFile[MAXSTRINGSIZE];    //!<file containing reconstructed noise model for `gb_catalog` to compute SNRs against.
    char cdfFile[MAXSTRINGSIZE];      //!<store `FILENAME` of input chain file from Flags::update.
    char gmmFile[MAXSTRINGSIZE];      //!<store `FILENAME` of input gmm file from Flags::update.
    char covFile[MAXSTRINGSIZE];      //!<store `FILENAME` of input covariance matrix file from Flags::updateCov.
    char matchInfile1[MAXSTRINGSIZE]; //!<input waveform \f$A\f$ for computing match \f$(h_A|h_B)\f$
    char matchInfile2[MAXSTRINGSIZE]; //!<input waveform \f$B\f$ for computing match \f$(h_A|h_B)\f$
    char pdfFile[MAXSTRINGSIZE];      //!<store `FILENAME` of input priors for Flags:knownSource.
    char psdFile[MAXSTRINGSIZE];      //!<store `FILENAME` of input psd file from Flags::psd.
    char catalogFile[MAXSTRINGSIZE];  //!<store `FILENAME` containing previously identified detections from Flags::catalog for cleaning padding regions
     
    int no_mbh;
    int no_ucb;
    int no_ucb_hi;
    int no_vgb;
    int no_noise;
    
    int qpad;
    double fmin;
    double fmax;

    char format[16];
    char dataDir[MAXSTRINGSIZE]; 

    int N;        //!<number of frequency bins
    int Nchannel; //!<number of data channels
    
    long cseed; //!<seed for MCMC set by `[--chainseed=INT; default=150914]`
    long nseed; //!<seed for noise realization set by `[--noiseseed=INT; default=151226]`
    long iseed; //!<seed for injection parameters set by `[--injseed=INT; default=151012]`
   
    double T; //!<observation time
    double t0;   //!<start times of segments
    
    int Nwave; //!<Number of samples for computing posterior reconstructions
    int downsample; //!<Downsample factor for getting the desired number of samples
    char fileName[MAXSTRINGSIZE]; //!<place holder for filnames
  
    struct GlobalFitData *global_fit;
    struct UCBData *ucb_data;
};

// inline void run_ucb_mcmc(struct Data *data, struct Orbit *orbit, struct Flags *flags, struct Chain *chain, struct Source *inj);
void glass_wrapper(struct Translator *translator);
void setup_ucb_global_fit(struct Translator *translator, int procID, int procID_min, int procID_max);
void clear_ucb_global_fit(struct Translator *translator, int procID, int procID_min, int procID_max);

#endif // mix_glass_h