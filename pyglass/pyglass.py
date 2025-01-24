from __future__ import annotations
from dataclasses import dataclass
from typing import List

from pyglass.translateglass import GlassGlobalFitTranslate

from eryn.moves import MHMove
from eryn.state import State

import numpy as np

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
    grid        : int =1
    update      : int =0
    updateCov   : int =0
    match       : int =0
    DMAX        : int =20


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
    ucbGridFile : str = "ucb_frequency_spacing.dat"  #!<`[--ucb-grid=FILENAME]` frequency grid for multiband UCB analysis
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


class GlassUCBGlobalFitMove(MHMove):

    def __init__(self, branch_name: str, glass_settings: GlassSettings, rank: int, min_ucb_rank: int, max_ucb_rank: int, *args, num_python_steps: int=1, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.branch_name = branch_name
        self.glass_settings = glass_settings
        self.gf = GlassGlobalFitTranslate(rank, min_ucb_rank, max_ucb_rank)
        self.gf.setup_ucb_global_fit(glass_settings)
        self.num_python_steps = num_python_steps

    def get_glass_state_as_eryn_state(self) -> State:
        (
            params,
            nleaves,
            logl,
            logp,
            betas
        ) = self.gf.get_current_glass_params()

        ntemps, nleaves_max, ndim = params.shape
        new_inds = self.convert_nleaves_to_inds(nleaves, nleaves_max)
        
        state = State(
            {self.branch_name: params[:, None, :, :]},
            inds={self.branch_name: new_inds[:, None, :]},
            log_like=logl[:, None],
            log_prior=logp[:, None],
            betas=betas,
            copy=True
        )
        return state

    def convert_nleaves_to_inds(self, nleaves: np.ndarray, nleaves_max: int) -> np.ndarray:
        ntemps = nleaves.shape[0]
        new_inds = np.zeros((ntemps, nleaves_max), dtype=bool)
        temp_map = np.tile(np.arange(nleaves_max), (ntemps, 1))
        new_inds[temp_map < nleaves[:, None]] = True
        return new_inds

    def fill_state_with_glass_state(self, state: State):
        (
            params,
            nleaves,
            logl,
            logp,
            betas
        ) = self.gf.get_current_glass_params()

        ntemps, nleaves_max, ndim = params.shape
        assert state.log_like.shape == (ntemps, 1)
        assert state.log_prior.shape == (ntemps, 1)
        assert state.betas.shape == (ntemps,)
        assert (ntemps, 1, nleaves_max, ndim) == state.branches[self.branch_name].shape
        
        this_branch = state.branches[self.branch_name]
        this_branch.coords[:] = params[:, None, :, :]
        state.log_like[:] = logl[:, None]
        state.log_prior[:] = logp[:, None]
        state.betas[:] = betas[:]

        # need to handle inds
        new_inds = self.convert_nleaves_to_inds(nleaves, nleaves_max)
        this_branch.inds[:] = new_inds[:, None, :]

    def set_glass_state_with_state(self, state: State):

        this_branch = state.branches[self.branch_name]
        logl = state.log_like[:, 0].copy()
        logp = state.log_prior[:, 0].copy()
        betas = state.betas[:].copy()
        ntemps, _, nleaves_max, ndim = this_branch.shape
        # need to handle inds
        # assumes they are not necesarrily comming in stacked against left side of array 
        # there may be gaps where inds == False
        # need to re-sort params order based on inds if gaps
        current_inds = this_branch.inds[:, 0].copy()
        nleaves = current_inds.sum(axis=-1).astype(np.int32).copy()
        temp_map = np.tile(np.arange(nleaves_max), (ntemps, 1))

        params = np.zeros((ntemps, nleaves_max, ndim))
        # need to sort going into glass
        # if no sorted is needed, it will return the input array
        params[temp_map < nleaves[:, None]] = this_branch.coords[this_branch.inds]
        params[temp_map >= nleaves[:, None]] = this_branch.coords[~this_branch.inds]
        
        self.gf.set_current_glass_params(params.flatten().copy(), nleaves.copy(), logl.copy(), logp.copy(), betas.copy())
        # TODO: this is temporary
        self.main_data = self.gf.get_main_tdi_data_in_glass()
        self.residual = self.gf.get_current_cold_chain_glass_residual()
        self.psd = self.gf.get_psd_in_glass()

    def propose(self, model, state: State) -> state:
        self.set_glass_state_with_state(state)

        # TODO: make data and psd come to from state
        self.gf.set_main_tdi_data_in_glass(self.main_data)
        self.gf.set_psd_in_glass(self.psd)

        for step in range(self.num_python_steps):
            self.gf.run_glass_ucb_step(step)
        
        new_state = State(state, copy=True)
        
        self.fill_state_with_glass_state(new_state)
        self.main_data[:] = self.gf.get_main_tdi_data_in_glass()
        self.residual[:] = self.gf.get_current_cold_chain_glass_residual()
        self.psd[:] = self.gf.get_psd_in_glass()

        accepted = np.zeros_like(state.log_like, dtype=int)
        self.temperature_control.swaps_accepted = np.zeros(state.betas.shape[0] - 1, dtype=int)
        self.temperature_control.swaps_proposed = np.zeros(state.betas.shape[0] - 1, dtype=int)
        print("nleaves:", state.branches["gb"].nleaves, "log like:", state.log_like[:, 0])
        return new_state, accepted