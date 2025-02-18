from __future__ import annotations
from dataclasses import dataclass
from typing import List

from pyglass.translateglass import GlassGlobalFitTranslate
from eryn.moves import MHMove
from eryn.state import State, BranchSupplemental

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
    grid        : int =0
    update      : int =0
    updateCov   : int =0
    match       : int =0
    DMAX        : int =20
    nUCB        : int =3

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


def glass_setup_ucb_rank(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
    assert comm is not None

    # get current rank and get index into class_ranks_list
    print(f"INSIDE GB search, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    gather_rank = class_ranks_list[0]
    assert gather_rank == min(class_ranks_list)
    min_ucb_rank = gather_rank
    max_ucb_rank = max(class_ranks_list)

    if isinstance(curr, GlassSettings):
        nUCB = curr.nUCB
        glass_set = curr
    else:
        raise NotImplementedError

    gf = GlassGlobalFitTranslate(rank, min_ucb_rank, max_ucb_rank, nUCB)
    gf.setup_ucb_global_fit_child(glass_set)
    gf.mpi_process_runner()


class GlassUCBGlobalFitMove(MHMove):

    def __init__(self, branch_names: str, glass_settings: GlassSettings, rank: int, min_ucb_rank: int, max_ucb_rank: int, *args, num_python_steps: int=1, nUCB: int=None, ranks_needed: int=1, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.branch_names = branch_names
        self.glass_settings = glass_settings
        if nUCB is None:
            raise ValueError("Must provide nUCB kwarg.")

        self.gf = GlassGlobalFitTranslate(rank, min_ucb_rank, max_ucb_rank, nUCB)
        self.gf.setup_ucb_global_fit_main(glass_settings)
        self.num_python_steps = num_python_steps
        self.ranks_needed = ranks_needed

        self.main_data = self.gf.get_main_tdi_data_in_glass()
        # self.residual = self.gf.get_current_cold_chain_glass_residual()
        self.psd = self.gf.get_psd_in_glass()
        
    @property
    def glass_df(self):
        return self.gf.df

    @property
    def glass_N(self):
        return self.gf.N
    
    @staticmethod
    def get_rank_function():
        return glass_setup_ucb_rank

    def get_glass_state_as_eryn_state(self) -> State:

        coords = {}
        inds = {}
        branch_supps = {}
        logl_all = []
        logp_all = []
        betas_all = []
        nleaves_all = []
        for i, key in enumerate(self.branch_names):
            (
                params,
                nleaves,
                logl,
                logp,
                betas
            ) = self.gf.get_current_glass_params(i)

            ntemps, nleaves_max, ndim = params.shape
            new_inds = self.convert_nleaves_to_inds(nleaves, nleaves_max)
            coords[key] = params[:, None, :, :]
            inds[key] = new_inds[:, None, :]
            logl_all.append(logl)
            logp_all.append(logp)
            betas_all.append(betas)
            nleaves_all.append(inds[key].sum(axis=(1, 2)))

        supp = BranchSupplemental({
            "logl": np.asarray(logl_all).T[:, None, :], 
            "logp": np.asarray(logp_all).T[:, None, :],
            "betas": np.asarray(betas_all).T[:, None, :],
            "nleaves": np.asarray(nleaves_all).T[:, None, :]
        }, base_shape=(ntemps, 1), copy=True)
            
        state = State(
            coords,
            inds=inds,
            log_like=np.zeros((ntemps, 1)),
            log_prior=np.zeros((ntemps, 1)),
            betas=np.zeros((ntemps,)),
            supplemental=supp,
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
        
        logl_all = []
        logp_all = []
        betas_all = []
        nleaves_all = []
        for i, key in enumerate(self.branch_names):
            (
                params,
                nleaves,
                logl,
                logp,
                betas
            ) = self.gf.get_current_glass_params(i)

            ntemps, nleaves_max, ndim = params.shape
            new_inds = self.convert_nleaves_to_inds(nleaves, nleaves_max)
            
            state.branches[key].coords[:] = params[:, None, :, :]
            state.branches[key].inds[:] = new_inds[:, None, :]
            logl_all.append(logl)
            logp_all.append(logp)
            betas_all.append(betas)
            nleaves_all.append(inds[key].sum(axis=(1, 2)))

        new_supp = BranchSupplemental({
            "logl": np.asarray(logl_all).T[:, None, :], 
            "logp": np.asarray(logp_all).T[:, None, :],
            "betas": np.asarray(betas_all).T[:, None, :],
            "nleaves": np.asarray(nleaves_all).T[:, None, :]
        }, base_shape=(ntemps, 1), copy=True)
            
        state.supplemental[:] = new_supp[:]
        
    def set_glass_state_with_state(self, state: State):

        for i, key in enumerate(self.branch_names):
            this_branch = state.branches[key]
            
            # get sub-band info
            tmp = state.supplemental[:, 0, i]
            logl = tmp["logl"].copy()
            logp = tmp["logp"].copy()
            betas = tmp["betas"].copy()
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
        
            self.gf.set_current_glass_params(i, params.flatten().copy(), nleaves.copy(), logl.copy(), logp.copy(), betas.copy())
            # TODO: this is temporary

    @property
    def nUCB(self):
        return self.gf.nUCB
        
    def propose(self, model: GlobalFitInfo, state: State) -> State:
        
        self.set_glass_state_with_state(state)

        # TODO: make data and psd come to from state
        self.gf.set_main_tdi_data_in_glass(model.analysis_container_arr[0].data_res_arr[:])
        self.gf.set_psd_in_glass(model.analysis_container_arr[0].sens_mat[:])

        for step in range(self.num_python_steps):
            self.gf.run_glass_ucb_step(step)
            print(f"\n\n\n\n STEP {step} complete.\n\n\n")

        new_state = State(state, copy=True)
        
        self.fill_state_with_glass_state(new_state)
        
        # self.residual[:] = self.gf.get_current_cold_chain_glass_residual()
        
        accepted = np.zeros_like(state.log_like, dtype=int)
        self.temperature_control.swaps_accepted = np.zeros(state.betas.shape[0] - 1, dtype=int)
        self.temperature_control.swaps_proposed = np.zeros(state.betas.shape[0] - 1, dtype=int)
        print("nleaves:", state.branches["gb"].nleaves, "log like:", state.log_like[:, 0])
        return new_state, accepted