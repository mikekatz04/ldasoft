#include "mixglass.h"
#include "stdio.h"
#include "stdlib.h"
#include "glass_data.h"
#include "ucb_mcmc_base.h"
#include "glass_ucb_wrapper.h"
#include "globalfit.h"

#include "mpi.h"

// #include "ucb_mcmc_base.h"

void parse_from_python(struct Translator *translator, struct Data *data, struct Orbit *orbit, struct Flags *flags, struct Chain *chain)
{
    //copy argv since getopt permutes order
    //Set defaults
    flags->rj          = translator->rj;
    flags->help        = translator->help;
    flags->calibration = translator->calibration;
    flags->verbose     = translator->verbose;
    flags->quiet       = translator->quiet;
    flags->simNoise    = translator->simNoise;
    flags->confNoise   = translator->confNoise;
    flags->burnin      = translator->burnin;
    flags->debug       = translator->debug;
    flags->strainData  = translator->strainData;
    flags->hdf5Data    = translator->hdf5Data;
    flags->psd         = translator->psd;
    flags->orbit       = translator->orbit;
    flags->prior       = translator->prior;
    flags->resume      = translator->resume;
    flags->NMCMC       = translator->NMCMC;
    flags->NBURN       = translator->NBURN;
    flags->threads     = translator->threads;  // omp_get_max_threads();
    memcpy(&flags->runDir, &translator->runDir, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->vbFile, &translator->vbFile, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->ucbGridFile, &translator->ucbGridFile, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->noiseFile, &translator->noiseFile, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->cdfFile, &translator->cdfFile, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->gmmFile, &translator->gmmFile, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->covFile, &translator->covFile, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->pdfFile, &translator->pdfFile, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->psdFile, &translator->psdFile, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->catalogFile, &translator->catalogFile, MAXSTRINGSIZE * sizeof(char));

    chain->NC          = translator->NC;//number of chains
    int set_fmax_flag  = 0; //flag watching for if fmax is set by CLI
    
    /* Simulated data building blocks */
    flags->no_mbh = translator->no_mbh;
    flags->no_ucb = translator->no_ucb;
    flags->no_ucb_hi = translator->no_ucb_hi;
    flags->no_vgb = translator->no_vgb;
    flags->no_noise = translator->no_noise;
    
    /*
     default data format is 'phase'
     optional support for 'frequency' a la LDCs
     */
    memcpy(&data->format, &translator->format, 16 * sizeof(char));
    memcpy(&data->fileName, &translator->fileName, MAXSTRINGSIZE * sizeof(char));
    
    data->T        = translator->T; /* one "mldc years" at 15s sampling */
    data->t0       = translator->t0; /* start time of data segment in seconds */
    data->sqT      = sqrt(data->T);
    data->qpad = translator->qpad;
    data->fmin = translator->fmin;
    data->fmin = translator->fmax;
    data->N        = translator->N;
    data->Nchannel = translator->Nchannel; //1=X, 2=AE, 3=XYZ
    
    data->cseed = translator->cseed;
    data->nseed = translator->nseed;
    data->iseed = translator->iseed;
    
    //Chains should be a multiple of threads for best usage of cores
    if(chain->NC % flags->threads !=0){
        chain->NC += flags->threads - (chain->NC % flags->threads);
    }
    //override size of data if fmax was requested
    if(set_fmax_flag) data->N = (int)floor((data->fmax - data->fmin)*data->T);
    
    //pad data
    data->N += 2*data->qpad;
    data->fmin -= data->qpad/data->T;
        
    //map fmin to nearest bin
    data->fmin = floor(data->fmin*data->T)/data->T;
    data->fmax = data->fmin + (double)data->N/data->T;

    //calculate helper quantities for likelihood normalizations
    data->logfmin   = log(data->fmin);
    data->sum_log_f = 0.0;
    for(int n=0; n<data->N; n++)
    {
        data->sum_log_f += log(data->fmin + (double)n/data->T);
    }
    
    ///// UCB PARSING

    //Set defaults
    flags->NINJ        = translator->NINJ;
    flags->fixSky      = translator->fixSky;
    flags->fixFreq     = translator->fixFreq;
    flags->fixFdot     = translator->fixFdot;
    flags->galaxyPrior = translator->galaxyPrior;
    flags->snrPrior    = translator->snrPrior;
    flags->emPrior     = translator->emPrior;
    flags->cheat       = translator->cheat;
    flags->detached    = translator->detached;
    flags->knownSource = translator->knownSource;
    flags->catalog     = translator->catalog;
    flags->grid        = translator->grid;
    flags->update      = translator->update;
    flags->updateCov   = translator->updateCov;
    flags->match       = translator->match;
    flags->DMAX        = translator->DMAX;

    memcpy(&flags->matchInfile1, &translator->matchInfile1, MAXSTRINGSIZE * sizeof(char));
    memcpy(&flags->matchInfile2, &translator->matchInfile2, MAXSTRINGSIZE * sizeof(char));
    
    flags->injFile = (char**)malloc(flags->DMAX *sizeof(char *));
    for(int n=0; n<flags->DMAX ; n++) flags->injFile[n] = (char*)malloc(1024*sizeof(char));
    
    for (int n=0; n<flags->NINJ; n++) memcpy(flags->injFile[n], translator->injFile[n], MAXSTRINGSIZE * sizeof(char));
}

void glass_wrapper(struct Translator *translator)
{
    // struct Flags * flags = (struct Flags*)flags_;
    printf("checkit: %d %s\n", translator->NC, translator->catalogFile);

    /* Allocate data structures */
    struct Flags  *flags = (struct Flags*)malloc(sizeof(struct Flags));
    struct Orbit  *orbit = (struct Orbit*)malloc(sizeof(struct Orbit));
    struct Chain  *chain = (struct Chain*)malloc(sizeof(struct Chain));
    struct Data   *data  = (struct Data*)malloc(sizeof(struct Data));
    struct Source *inj   = (struct Source*)malloc(sizeof(struct Source));

    parse_from_python(translator, data, orbit, flags, chain);

    printf("checkit: %d %d %d\n", data->N, translator->N, data->qpad);

    run_ucb_mcmc(data, orbit, flags, chain, inj);

    free(flags);
    free(orbit);
    free(chain);
    free(data);
    free(inj);
}


void clear_ucb_global_fit(struct Translator *translator, int procID, int procID_min, int procID_max)
{
    dealloc_ucb_state_python(translator->ucb_data->model, translator->ucb_data->trial, translator->ucb_data->flags, translator->ucb_data->chain);
    
    free_data(translator->ucb_data->data);

    dealloc_gf_data(translator->global_fit);
    dealloc_ucb_data(translator->ucb_data);

    free(translator->global_fit);
    free(translator->ucb_data);

    int finalized;
    // You also need this when your program is about to exit
    MPI_Finalized(&finalized);
    if (!finalized)
    MPI_Finalize();
}

void my_bcast(void* data, int count, MPI_Datatype datatype, int root,
              MPI_Comm communicator, int procID, int procID_min, int procID_max) {

   int world_size = procID_max - procID_min + 1;

   if (procID == root) {
    // If we are the root process, send our data to everyone
    int i, temp_procID;
    for (i = 0; i < world_size; i++) {
        temp_procID = procID_min + i;
      if (temp_procID != root) {
        MPI_Send(data, count, datatype, temp_procID, 0, communicator);
      }
    }
  } else {
    // If we are a receiver process, receive the data from the root
    MPI_Recv(data, count, datatype, root, 0, communicator,
             MPI_STATUS_IGNORE);
  }
}


void share_data_mix(struct TDI *tdi_full, int root, int procID, int procID_min, int procID_max)
{
    //first tell all processes how large the dataset is
    my_bcast(&tdi_full->N, 1, MPI_INT, root, MPI_COMM_WORLD, procID, procID_min, procID_max);
    //all but root process need to allocate memory for TDI structure
    if(procID!=root) alloc_tdi(tdi_full, tdi_full->N, N_TDI_CHANNELS);
    
    //now broadcast contents of TDI structure
    my_bcast(&tdi_full->delta, 1, MPI_DOUBLE, root, MPI_COMM_WORLD, procID, procID_min, procID_max);
    my_bcast(tdi_full->X, 2*tdi_full->N, MPI_DOUBLE, root, MPI_COMM_WORLD, procID, procID_min, procID_max);
    my_bcast(tdi_full->Y, 2*tdi_full->N, MPI_DOUBLE, root, MPI_COMM_WORLD, procID, procID_min, procID_max);
    my_bcast(tdi_full->Z, 2*tdi_full->N, MPI_DOUBLE, root, MPI_COMM_WORLD, procID, procID_min, procID_max);
    my_bcast(tdi_full->A, 2*tdi_full->N, MPI_DOUBLE, root, MPI_COMM_WORLD, procID, procID_min, procID_max);
    my_bcast(tdi_full->E, 2*tdi_full->N, MPI_DOUBLE, root, MPI_COMM_WORLD, procID, procID_min, procID_max);
    my_bcast(tdi_full->T, 2*tdi_full->N, MPI_DOUBLE, root, MPI_COMM_WORLD, procID, procID_min, procID_max);
    
}

void dealloc_ucb_state_python(struct Model **model, struct Model **trial, struct Flags *flags, struct Chain *chain)
{
    int NC = chain->NC;
    int DMAX = flags->DMAX;
    for(int ic=0; ic<NC; ic++)
    {   
        free_model(trial[ic]);
        free_model(model[ic]);     
    }//end loop over chains
    free(trial);
    free(model);
}


// void init_chains(struct Data *data, struct Orbit *orbit, struct Flags *flags, struct Chain *chain, struct Proposal **proposal, struct Model **model, struct Model **trial)
// {
//     int NC = chain->NC;
//     int DMAX = flags->DMAX;
//     for(int ic=0; ic<NC; ic++)
//     {
//         //set noise model
//         copy_noise(data->noise, model[ic]->noise);
        
//         //draw signal model
//         for(int n=0; n<DMAX; n++)
//         {
//             //map parameters to vector
//             model[ic]->source[n]->f0       = inj->f0;
//             model[ic]->source[n]->dfdt     = inj->dfdt;
//             model[ic]->source[n]->costheta = inj->costheta;
//             model[ic]->source[n]->phi      = inj->phi;
//             model[ic]->source[n]->amp      = inj->amp;
//             model[ic]->source[n]->cosi     = inj->cosi;
//             model[ic]->source[n]->phi0     = inj->phi0;
//             model[ic]->source[n]->psi      = inj->psi;
//             model[ic]->source[n]->d2fdt2   = inj->d2fdt2;
//             map_params_to_array(model[ic]->source[n], model[ic]->source[n]->params, data->T);     
//         }
//         else
//         {
//             draw_from_uniform_prior(data, model[ic], model[ic]->source[n], proposal[0], model[ic]->source[n]->params , chain->r[ic]);
//         }
//         map_array_to_params(model[ic]->source[n], model[ic]->source[n]->params, data->T);
//         galactic_binary_fisher(orbit, data, model[ic]->source[n], data->noise);
//         model[ic]->source[n]->fisher_update_flag=0;
             
//         // Form master model & compute likelihood of starting position
//         generate_noise_model(data, model[ic]);
//         generate_signal_model(orbit, data, model[ic], -1);
        
//         //calibration error
//         if(flags->calibration)
//         {
//             draw_calibration_parameters(data, model[ic], chain->r[ic]);
//             generate_calibration_model(data, model[ic]);
//             apply_calibration_model(data, model[ic]);
//         }
    
//         if(!flags->prior)
//         {
//             model[ic]->logL     = gaussian_log_likelihood(data, model[ic]);
//             model[ic]->logLnorm = gaussian_log_likelihood_constant_norm(data, model[ic]);
//         }
//         else model[ic]->logL = model[ic]->logLnorm = 0.0;
        
//         if(ic==0) chain->logLmax += model[ic]->logL + model[ic]->logLnorm;
//     }
// }




void setup_ucb_global_fit(struct Translator *translator, int procID, int procID_min, int procID_max)
{
    int color = 0;

    // Is this okay?
    int root = procID_min;
    int nprocs = procID_max - procID_min + 1;
    int initialized;

    MPI_Initialized(&initialized);
    if (!initialized)
    {
        MPI_Init(NULL, NULL);
    }

    int procID_here, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &procID_here);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Allocate structures to hold global model */
    translator->global_fit = malloc(sizeof(struct GlobalFitData));
    translator->ucb_data   = malloc(sizeof(struct UCBData));
    
    struct GlobalFitData *global_fit = translator->global_fit;
    struct UCBData       *ucb_data = translator->ucb_data;
    
    alloc_gf_data(global_fit);

    /* Allocate UCB data structures */
    alloc_ucb_data(ucb_data, procID);
  
    /* Aliases to ucb structures */
    struct Flags *flags = ucb_data->flags;
    struct Orbit *orbit = ucb_data->orbit;
    struct Chain *chain = ucb_data->chain;
    struct Data  *data  = ucb_data->data;

    /* all processes parse command line and set defaults/flags */
    parse_from_python(translator, data, orbit, flags, chain);
    //if(procID==0 && flags->help) print_usage();
      
    /* Store size of each model component */
    global_fit->nUCB = procID_max - procID_min + 1; //room for noise model

    /* Assign processes to models */
    
    //ucb model takes remaining nodes
    ucb_data->procID_min = procID_min;
    ucb_data->procID_max = procID_max;

    setup_frequency_segment(ucb_data);
    
    /* Initialize ucb data structures */
    alloc_data(data, flags);

    /* allocate global fit noise model for all processes */
    // I REMOVED noise_data-> here, is that okay????
    alloc_noise(global_fit->psd, data->N, data->Nchannel);

     /* root process reads data */
    // confirm processes are the same
    
    if(procID==root) ReadHDF5(data,global_fit->tdi_full,flags);
    // MPI_Barrier(MPI_COMM_WORLD);
    
    
    /* alias of full TDI data */
    struct TDI *tdi_full = global_fit->tdi_full;
    
    
    /* broadcast data to all ucb processes and select frequency segment */
    share_data_mix(tdi_full, root, procID, procID_min, procID_max);
    
    /* composite models are allocated to be the same size as tdi_full */
    setup_gf_data(global_fit);

    /* set up data for ucb model processes */
    setup_ucb_data(ucb_data, tdi_full);
    
    initialize_ucb_sampler(ucb_data);

    copy_noise(ucb_data->data->noise,global_fit->psd);
    // TODO: NEEEEEDDD TOO CHECK THIS
    printf("%e %e\n", global_fit->psd->C[0][0][100], global_fit->psd->C[1][1][200]);
    printf("Finished ucb global fit setup\n");
}

void run_glass_ucb_step(int step, struct Translator *translator, int procID, int procID_min, int procID_max)
{

    struct GlobalFitData *global_fit = translator->global_fit;
    struct UCBData       *ucb_data = translator->ucb_data;
    int VGB_Flag = 0;
    int MBH_Flag = 0;
    int UCB_Flag = 1;
    run_ucb_update(procID, global_fit, ucb_data, UCB_Flag, VGB_Flag, MBH_Flag);

    if (step % 100 == 0)
        printf("step: %d, nmax: %d, neff: %d,  nlive: %d, logL: %e\n", step, ucb_data->model[0]->Nmax, ucb_data->model[0]->Neff, ucb_data->model[0]->Nlive, ucb_data->model[0]->logL);
}
    
