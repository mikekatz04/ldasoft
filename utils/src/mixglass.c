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
    // printf("data->NChannel: %d\n", data->Nchannel);
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
    // printf("checkit: %d %s\n", translator->NC, translator->catalogFile);

    /* Allocate data structures */
    struct Flags  *flags = (struct Flags*)malloc(sizeof(struct Flags));
    struct Orbit  *orbit = (struct Orbit*)malloc(sizeof(struct Orbit));
    struct Chain  *chain = (struct Chain*)malloc(sizeof(struct Chain));
    struct Data   *data  = (struct Data*)malloc(sizeof(struct Data));
    struct Source *inj   = (struct Source*)malloc(sizeof(struct Source));

    parse_from_python(translator, data, orbit, flags, chain);

    // printf("checkit: %d %d %d\n", data->N, translator->N, data->qpad);

    run_ucb_mcmc(data, orbit, flags, chain, inj);

    free(flags);
    free(orbit);
    free(chain);
    free(data);
    free(inj);
}


void clear_ucb_global_fit_main(struct Translator *translator, int procID, int procID_min, int procID_max)
{
    dealloc_ucb_state_python(translator->ucb_data->model, translator->ucb_data->trial, translator->ucb_data->flags, translator->ucb_data->chain);
    
    for (int i = 0; i < translator->global_fit->nUCB; i += 1)
    {
        free_data(translator->ucb_data_all[i]->data);
        dealloc_ucb_data(translator->ucb_data_all[i]);
        free(translator->ucb_data_all[i]);
    }
    dealloc_gf_data(translator->global_fit);
    free(translator->global_fit);

    int finalized;
    // You also need this when your program is about to exit
    MPI_Finalized(&finalized);
    if (!finalized)
    MPI_Finalize();
}

void clear_ucb_global_fit_child(struct Translator *translator, int procID, int procID_min, int procID_max)
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



void setup_ucb_global_fit_main(struct Translator *translator, int procID, int procID_min, int procID_max, int nUCB)
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
    translator->global_fit = (struct GlobalFitData*)malloc(sizeof(struct GlobalFitData));
    // translator->ucb_data   = ((struct UCBData)*)malloc(sizeof(struct UCBData));
    translator->ucb_data_all = (struct UCBData**)malloc(nUCB * sizeof(struct UCBData));
    
    struct GlobalFitData *global_fit = translator->global_fit;
    // struct UCBData       *ucb_data = translator->ucb_data;
    struct UCBData      **ucb_data_all = translator->ucb_data_all;
    global_fit->nUCB = nUCB; // procID_max - procID_min + 1; //room for noise model
    alloc_gf_data(global_fit);

    struct UCBData       *ucb_data;

    for (int i = 0; i < nUCB; i += 1)
    {
        ucb_data_all[i] = (struct UCBData*)malloc(sizeof(struct UCBData));
        ucb_data = ucb_data_all[i];
        
        /* Allocate UCB data structures */
        alloc_ucb_data(ucb_data, procID);

        /* Aliases to ucb structures */
        struct Flags *flags = ucb_data->flags;
        struct Orbit *orbit = ucb_data->orbit;
        struct Chain *chain = ucb_data->chain;
        struct Data  *data  = ucb_data->data;
        struct Data  tmp_data; 
        /* all processes parse command line and set defaults/flags */
        parse_from_python(translator, data, orbit, flags, chain);
        //if(procID==0 && flags->help) print_usage();
        // printf("CHECKECHECKCHECK: %d\n", data->Nchannel);
        /* Store size of each model component */
        
        /* Assign processes to models */
        
        //ucb model takes remaining nodes
        ucb_data->procID_min = procID_min;
        ucb_data->procID_max = procID_max;

        setup_frequency_segment(ucb_data);
        // ("CHECKECHECKCHECK22: %d\n", ucb_data->data->Nchannel);
        
        /* Initialize ucb data structures */
        alloc_data(data, flags);
        //printf("CHECKECHECKCHECK22: %d\n", data->Nchannel);
        
        if (i == 0)
        {
            ReadHDF5(data,global_fit->tdi_full,flags);
            // MLK: need this here to get number before alloc_noise
            global_fit->tdi_full->Nchannel = data->Nchannel;
            
            /* composite models are allocated to be the same size as tdi_full */
            setup_gf_data(global_fit);

            /* allocate global fit noise model for all processes */
            // I REMOVED noise_data-> here, is that okay????
            alloc_noise(global_fit->psd, global_fit->tdi_full->N, global_fit->tdi_full->Nchannel);
            
        }
        initialize_orbit(data,orbit,flags);
        tmp_data.fmin = 0.0;
        tmp_data.N = global_fit->tdi_full->N;
        tmp_data.Nchannel = global_fit->tdi_full->Nchannel;
        tmp_data.noise = global_fit->psd;
        tmp_data.T = ucb_data->data->T;
        for (int i = 0; i < 16; i += 1)
        {
            tmp_data.format[i] = ucb_data->data->format[i];
        }
        GetNoiseModel(&tmp_data, orbit, flags);

        /* root process reads data */
        // confirm processes are the same
        
        
        // MPI_Barrier(MPI_COMM_WORLD);
        
        
        /* alias of full TDI data */
        struct TDI *tdi_full = global_fit->tdi_full;
        
        /* set up data for ucb model processes */
        setup_ucb_data(ucb_data, tdi_full);
        
        sprintf(ucb_data->data->dataDir,"%sdata",flags->runDir);
        sprintf(ucb_data->chain->chainDir,"%schains",flags->runDir);
        sprintf(ucb_data->chain->chkptDir,"%scheckpoint",flags->runDir);
        
        initialize_ucb_sampler(ucb_data);
        // copy_noise(ucb_data->data->noise,global_fit->psd);
        // TODO: NEEEEEDDD TOO CHECK THIS
    }
}

void setup_ucb_global_fit_child(struct Translator *translator, int procID, int procID_min, int procID_max, int nUCB)
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
    struct Data  tmp_data; 
    /* all processes parse command line and set defaults/flags */
    parse_from_python(translator, data, orbit, flags, chain);
    //if(procID==0 && flags->help) print_usage();
    // printf("CHECKECHECKCHECK: %d\n", data->Nchannel);
    /* Store size of each model component */
    global_fit->nUCB = procID_max - procID_min + 1; //room for noise model

    /* Assign processes to models */
    
    //ucb model takes remaining nodes
    ucb_data->procID_min = procID_min;
    ucb_data->procID_max = procID_max;

    setup_frequency_segment(ucb_data);
    // ("CHECKECHECKCHECK22: %d\n", ucb_data->data->Nchannel);
    
    /* Initialize ucb data structures */
    alloc_data(data, flags);
    //printf("CHECKECHECKCHECK22: %d\n", data->Nchannel);
    
    ReadHDF5(data,global_fit->tdi_full,flags);
    // MLK: need this here to get number before alloc_noise
    global_fit->tdi_full->Nchannel = data->Nchannel;
    
    
    /* allocate global fit noise model for all processes */
    // I REMOVED noise_data-> here, is that okay????
    alloc_noise(global_fit->psd, global_fit->tdi_full->N, global_fit->tdi_full->Nchannel);
    initialize_orbit(data,orbit,flags);
    tmp_data.fmin = 0.0;
    tmp_data.N = global_fit->tdi_full->N;
    tmp_data.Nchannel = global_fit->tdi_full->Nchannel;
    tmp_data.noise = global_fit->psd;
    tmp_data.T = ucb_data->data->T;
    for (int i = 0; i < 16; i += 1)
    {
        tmp_data.format[i] = ucb_data->data->format[i];
    }
    GetNoiseModel(&tmp_data, orbit, flags);

     /* root process reads data */
    // confirm processes are the same
    
    
    // MPI_Barrier(MPI_COMM_WORLD);
    
    
    /* alias of full TDI data */
    struct TDI *tdi_full = global_fit->tdi_full;
    
    
    /* broadcast data to all ucb processes and select frequency segment */
    // share_data_mix(tdi_full, root, procID, procID_min, procID_max);
    
    
    /* composite models are allocated to be the same size as tdi_full */
    setup_gf_data(global_fit);

    /* set up data for ucb model processes */
    setup_ucb_data(ucb_data, tdi_full);
    
    sprintf(ucb_data->data->dataDir,"%sdata",flags->runDir);
    sprintf(ucb_data->chain->chainDir,"%schains",flags->runDir);
    sprintf(ucb_data->chain->chkptDir,"%scheckpoint",flags->runDir);
    
    initialize_ucb_sampler(ucb_data);
    // copy_noise(ucb_data->data->noise,global_fit->psd);
    // TODO: NEEEEEDDD TOO CHECK THIS
}

// void clear_mpi_process(int procID, struct Translator *translator, int procID_target)
// {
//     int indicator = -1;
//     int procID_tmp = 0;
//     int action_type = 0;
//     int ucb_index_receive = 0;
//     MPI_Status status;
//     int ready = -1;
//     MPI_Recv(&procID_tmp, 1, MPI_INT, procID_target, 33, MPI_COMM_WORLD, &status);
//     MPI_Recv(&procID_tmp, 1, MPI_INT, procID_target, 1, MPI_COMM_WORLD, &status);
//     printf("CHECK1\n");
    
//     if (procID_target != procID_tmp)
//     {
//         fprintf(stderr, "Target proc and received proc must be the same. \n");
//         exit(-1);
//     }
//     printf("CHECK2\n");
//     MPI_Recv(&action_type, 1, MPI_INT, procID_target, 1, MPI_COMM_WORLD, &status);
    
//     printf("CHECK3 %d\n", action_type);
    
//     if (action_type != 2)
//     {
//         fprintf(stderr, "Action type not correct. \n");
//         exit(-1);
//     }
//     printf("CHECK4\n");
    
//     MPI_Send(&indicator, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD);
//     printf("CHECK5\n");
    
//     MPI_Recv(&ucb_index_receive, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD, &status);
//     printf("CHECK6\n");
    
//     mpi_process_runner_inner(procID, translator->ucb_data_all[ucb_index_receive], translator->global_fit, procID_target, false);
// }


void end_mpi_process(int procID, struct Translator *translator, int procID_target)
{
    int indicator = -1;
    int procID_tmp = 0;
    int action_type = 0;
    int ucb_index_receive = 0;
    MPI_Status status;
    MPI_Recv(&procID_tmp, 1, MPI_INT, procID_target, 1, MPI_COMM_WORLD, &status);
    if (procID_target != procID_tmp)
    {
        fprintf(stderr, "Target proc and received proc must be the same. \n");
        exit(-1);
    }
    MPI_Recv(&action_type, 1, MPI_INT, procID_target, 1, MPI_COMM_WORLD, &status);
    
    if (action_type != 1)
    {
        fprintf(stderr, "Action type not correct. \n");
        exit(-1);
    }
    printf("end indicator send %d\n", procID_target);
    indicator = -1;
    MPI_Send(&indicator, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD);
}

bool check_all_full(int *ucb_index_running, int nprocs)
{
    bool is_full = true;
    for (int i = 0; i < nprocs; i += 1)
    {
        if (ucb_index_running[i] == -1)
        {
            is_full = false;
            break;
        }
    }
    return is_full;
}

bool check_all_empty(int *ucb_index_running, int nprocs)
{
    bool is_empty = true;
    for (int i = 0; i < nprocs; i += 1)
    {
        if (ucb_index_running[i] != -1)
        {
            is_empty = false;
            break;
        }
    }
    return is_empty;
}

int get_next_free_process(int *ucb_index_running, int nprocs)
{
    int next_free_process = -1;
    for (int i = 0; i < nprocs; i += 1)
    {
        if (ucb_index_running[i] == -1)
        {
            next_free_process = i;
            break;
        }
    }
    return next_free_process;
}

void run_glass_ucb_step(int step, struct Translator *translator, int procID, int procID_min, int procID_max)
{

    int root = procID_min;
    int nprocs_extra = procID_max - procID_min; // +1 -1 there
    
    struct GlobalFitData *global_fit = translator->global_fit;
    struct UCBData       *ucb_data = translator->ucb_data;
    int VGB_Flag = 0;
    int MBH_Flag = 0;
    int UCB_Flag = 1;
    bool all_full = false;
    bool all_empty = false;
    int recv_indicator;
    int next_free_process;
    MPI_Status status;
    if (nprocs_extra > 0)
    {
        for (int base = 0; base < 2; base += 1)
        {
            translator->ucb_index_running = (int*) malloc(nprocs_extra * sizeof(int));
            for (int i = 0; i < nprocs_extra; i += 1) translator->ucb_index_running[i] = -1;
            for (int i = base; i < translator->global_fit->nUCB; i += 2)
            {
                printf("Start %d look: \n", i);
                for (int j = 0; j < nprocs_extra; j += 1) printf("%d, ", translator->ucb_index_running[j]);
                printf(".\n");
                all_full = check_all_full(translator->ucb_index_running, nprocs_extra);
                if (all_full)
                {
                    // remove next that is ready
                    printf("to clear\n");
                    clear_mpi_process(procID, translator);
                    all_full = check_all_full(translator->ucb_index_running, nprocs_extra);
                    if (all_full)
                    {
                        fprintf(stderr, "all_full issue\n");
                        exit(-1);
                    }
                }
                next_free_process = get_next_free_process(translator->ucb_index_running, nprocs_extra);
                printf("Next free process %d: \n", next_free_process);
                mpi_process_send_out(procID, translator, procID_min, next_free_process, i); // next_free_process starts at 0
            }
            printf("pre end look: \n");
            for (int j = 0; j < nprocs_extra; j += 1) printf("%d, ", translator->ucb_index_running[j]);
            printf(".\n");
            while(!check_all_empty(translator->ucb_index_running, nprocs_extra))
            {
                clear_mpi_process(procID, translator);
            }
            printf("end look: \n");
            for (int j = 0; j < nprocs_extra; j += 1) printf("%d, ", translator->ucb_index_running[j]);
            printf(".\n");

            // REBUILD GLOBAL DATA AND MOVE PARAMETERS
            
        }  
        
        free(translator->ucb_index_running);
    }
    else
    {
        for (int i = 0; i < translator->global_fit->nUCB; i += 1)
        {
            run_ucb_update(procID, translator->global_fit, translator->ucb_data_all[i], 1, 0, 0);
        }    
    }

    // if (step % 100 == 0)
    //     printf("step: %d, nmax: %d, neff: %d,  nlive: %d, logL: %e\n", step, ucb_data->model[0]->Nmax, ucb_data->model[0]->Neff, ucb_data->model[0]->Nlive, ucb_data->model[0]->logL);
}

void get_current_glass_params(struct Translator *translator, double *params, int *nleaves, double *logl, double *logp, double* betas, int array_length, int ucb_index)
{
    struct UCBData *ucb_data = translator->ucb_data_all[ucb_index];
    struct Chain *chain = ucb_data->chain;
    struct Model **model = ucb_data->model;
    if (array_length != chain->NC * translator->DMAX)
    {
        fprintf(stderr, "array_length (%d) != chains * DMAX (%d) \n", chain->NC * translator->DMAX);
        exit(-1);
    }

    int index_params = 0;
    int model_index = 0;
    double logPx = 0.0;
    for (int i = 0; i < chain->NC; i += 1)
    {
        model_index = chain->index[i];
        nleaves[i] = model[model_index]->Nlive;
        betas[i] = 1. / chain->temperature[i];
        logl[i] = model[model_index]->logL;
        logPx = 0.0;  
        for (int j = 0; j < translator->DMAX; j += 1)
        {
            //logPx += evaluate_prior(ucb_data->flags, ucb_data->data, model[i], model[i]->prior, model[i]->source[j]->params);
            
            for (int k = 0; k < UCB_MODEL_NP; k += 1)
            {
                index_params = (i * translator->DMAX + j) * UCB_MODEL_NP + k; 
                
                // if (j >= model[i]->Nlive)
                // {
                //     params[index_params] = 0.0;
                // }
                // else
                // {
                    params[index_params] = model[model_index]->source[j]->params[k];
                // {
                
                // if ((i < 2) && (j < 2) && (k < 2))
                //     printf("%d, %d, %d, %e\n", i, j, k, model[i]->source[j]->params[k]);
            }
        }
        //logp[i] = logPx;
    }
}
    

void set_current_glass_params(struct Translator *translator, double *params, int *nleaves, double *logl, double *logp, double *betas, int array_length, int ucb_index)
{
    struct UCBData *ucb_data = translator->ucb_data_all[ucb_index];
    struct Chain *chain = ucb_data->chain;
    struct Model **model = ucb_data->model;

    if (array_length != chain->NC * translator->DMAX)
    {
        fprintf(stderr, "array_length (%d) != chains * DMAX (%d) \n", chain->NC * translator->DMAX);
        exit(-1);
    }

    int index_params = 0;
    int model_index = 0;
    for (int i = 0; i < chain->NC; i += 1)
    {
        model_index = chain->index[i];
        model[model_index]->Nlive = nleaves[i];
        chain->temperature[i] = 1. / betas[i];
        model[model_index]->logL = logl[i];
        for (int j = 0; j < translator->DMAX; j += 1)
        {
            // if (j >= model[i]->Nlive) continue;
            
            for (int k = 0; k < UCB_MODEL_NP; k += 1)
            {
                index_params = (i * translator->DMAX + j) * UCB_MODEL_NP + k; 
                model[model_index]->source[j]->params[k] = params[index_params];
                // if ((i < 2) && (j < 2) && (k < 2))
                //     printf("set %d, %d, %d, %e\n", i, j, k, model[i]->source[j]->params[k]);
            }
        }
    }
}


void get_current_cold_chain_glass_residual(struct Translator *translator, double *data_arr, int Nchannel, int N)
{
    struct UCBData *ucb_data = translator->ucb_data;
    struct TDI *tdi = translator->global_fit->tdi_full;
    struct Chain *chain = ucb_data->chain;
    struct Data *data = ucb_data->data;
    struct Model *model = ucb_data->model[chain->index[0]];
    int N2 = tdi->N*2;
    if (N != tdi->N)
    {
        fprintf(stderr,"Input and output lengths are different. N in: %d ; N in c backend: %d\n", N, data->N);
        exit(1);
    }

    if (Nchannel != tdi->Nchannel)
    {
        fprintf(stderr,"Input and output number of channels are different. Nchannel in: %d ; Nchannel in c backend: %d\n", Nchannel, data->Nchannel);
        exit(1);
    }

    for(int i=0; i<N2; i++)
    {
        //printf("OEWFEOWJ: %e %e\n", model->noise->C[0][0][0], model->noise->C[1][1][0]);
                
        switch(tdi->Nchannel)
        {
            case 2:
                data_arr[0 * N2 + i] = tdi->A[i]; // - model->tdi->A[i];
                data_arr[1 * N2 + i] = tdi->E[i]; // - model->tdi->E[i];
                break;
            case 3:
                data_arr[0 * N2 + i] = tdi->X[i]; // - model->tdi->X[i];
                data_arr[1 * N2 + i] = tdi->Y[i]; // - model->tdi->Y[i];
                data_arr[2 * N2 + i] = tdi->Z[i]; // - model->tdi->Z[i];
                break;
            default:
                fprintf(stderr,"Unsupported number of channels in gaussian_log_likelihood()\n");
                exit(1);
        }
    }
}

int get_frequency_domain_data_length(struct Translator *translator)
{
    return translator->global_fit->tdi_full->N;
}

double get_frequency_domain_data_df(struct Translator *translator)
{
    return translator->global_fit->tdi_full->delta;
}

void get_main_tdi_data_in_glass(struct Translator *translator, double *data_arr, int Nchannel, int N)
{

    struct TDI *tdi = translator->global_fit->tdi_full;
    
    int N2 = tdi->N*2;
    if (N != tdi->N)
    {
        fprintf(stderr,"Input and output lengths are different. N in: %d ; N in c backend: %d\n", N, tdi->N);
        exit(1);
    }

    if (Nchannel != tdi->Nchannel)
    {
        fprintf(stderr,"Input and output number of channels are different. Nchannel in: %d ; Nchannel in c backend: %d\n", Nchannel, tdi->Nchannel);
        exit(1);
    }

    for(int i=0; i<N2; i++)
    {
        //printf("OEWFEOWJ: %e %e\n", model->noise->C[0][0][0], model->noise->C[1][1][0]);
                
        switch(tdi->Nchannel)
        {
            case 2:
                data_arr[0 * N2 + i] = tdi->A[i];
                data_arr[1 * N2 + i] = tdi->E[i];
                break;
            case 3:
                data_arr[0 * N2 + i] = tdi->X[i];
                data_arr[1 * N2 + i] = tdi->Y[i];
                data_arr[2 * N2 + i] = tdi->Z[i];
                break;
            default:
                fprintf(stderr,"Unsupported number of channels in gaussian_log_likelihood()\n");
                exit(1);
        }
    }
}


void set_main_tdi_data_in_glass(struct Translator *translator, double *data_arr, int Nchannel, int N)
{
    struct TDI *tdi = translator->global_fit->tdi_full;
    
    int N2 = tdi->N*2;
    if (N != tdi->N)
    {
        fprintf(stderr,"Input and output lengths are different. N in: %d ; N in c backend: %d\n", N, tdi->N);
        exit(1);
    }

    if (Nchannel != tdi->Nchannel)
    {
        fprintf(stderr,"Input and output number of channels are different. Nchannel in: %d ; Nchannel in c backend: %d\n", Nchannel, tdi->Nchannel);
        exit(1);
    }

    for(int i=0; i<N2; i++)
    {
        //printf("OEWFEOWJ: %e %e\n", model->noise->C[0][0][0], model->noise->C[1][1][0]);
                
        switch(tdi->Nchannel)
        {
            case 2:
                tdi->A[i] = data_arr[0 * N2 + i];
                tdi->E[i] = data_arr[1 * N2 + i];
                break;
            case 3:
                tdi->X[i] = data_arr[0 * N2 + i];
                tdi->Y[i] = data_arr[1 * N2 + i];
                tdi->Z[i] = data_arr[2 * N2 + i];
                break;
            default:
                fprintf(stderr,"Unsupported number of channels in gaussian_log_likelihood()\n");
                exit(1);
        }
    }
}


void get_psd_in_glass(struct Translator *translator, double *noise_arr, int Nchannel, int N)
{
    struct Noise *noise = translator->global_fit->psd;

    if (N != noise->N)
    {
        fprintf(stderr,"Input and output lengths are different. N in: %d ; N in c backend: %d\n", N, noise->N);
        exit(1);
    }

    if (Nchannel != noise->Nchannel)
    {
        fprintf(stderr,"Input and output number of channels are different. Nchannel in: %d ; Nchannel in c backend: %d\n", Nchannel, noise->Nchannel);
        exit(1);
    }

    for(int i=0; i<noise->N; i++)
    {
        //printf("OEWFEOWJ: %e %e\n", model->noise->C[0][0][0], model->noise->C[1][1][0]);
                
        switch(noise->Nchannel)
        {
            case 2:
                for (int j = 0; j < 2; j += 1)
                {
                    noise_arr[j * noise->N + i] = noise->C[j][j][i];
                }
                break;
            case 3:
                for (int j = 0; j < 3; j += 1)
                {
                    for (int k = 0; k < 3; k += 1)
                    {
                        noise_arr[(j * 3 + k) * noise->N + i] = noise->C[j][k][i];
                    }
                }
                break;
            default:
                fprintf(stderr,"Unsupported number of channels in gaussian_log_likelihood()\n");
                exit(1);
        }
    }
}


void set_psd_in_glass(struct Translator *translator, double *noise_arr, int Nchannel, int N)
{
    struct Noise *noise = translator->global_fit->psd;
    
    if (N != noise->N)
    {
        fprintf(stderr,"Input and output lengths are different. N in: %d ; N in c backend: %d\n", N, noise->N);
        exit(1);
    }

    if (Nchannel != noise->Nchannel)
    {
        fprintf(stderr,"Input and output number of channels are different. Nchannel in: %d ; Nchannel in c backend: %d\n", Nchannel, noise->Nchannel);
        exit(1);
    }

    for(int i=0; i<noise->N; i++)
    {
        //printf("OEWFEOWJ: %e %e\n", model->noise->C[0][0][0], model->noise->C[1][1][0]);
                
        switch(noise->Nchannel)
        {
            case 2:
                for (int j = 0; j < 2; j += 1)
                {
                    noise->C[j][j][i] = noise_arr[j * noise->N + i];
                }
                break;
            case 3:
                for (int j = 0; j < 3; j += 1)
                {
                    for (int k = 0; k < 3; k += 1)
                    {
                        noise->C[j][j][i] = noise_arr[j * noise->N + i];
                    }
                }
                break;
            default:
                fprintf(stderr,"Unsupported number of channels in gaussian_log_likelihood()\n");
                exit(1);
        }
    }
}



void mpi_send_chain(struct Chain *chain, int receiver)
{
    /// Number of chains
    MPI_Send(&chain->NC, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    
    /// Array containing current order of chains in temperature ladder
    MPI_Send(chain->index, chain->NC, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    
    /// Size of model being sampled by chain.  Depricated?
    //int **dimension;
    
    /// Array tracking acceptance rate of parallel chain exchanges.
    MPI_Send(chain->acceptance, chain->NC, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    
    /// Array of chain temperatures
    MPI_Send(chain->temperature, chain->NC, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    /// Array storing \f$\langle \log p(d|\vec\theta)^{1/T} \rangle\f$ for thermodynamic integration.
    MPI_Send(chain->avgLogL, chain->NC, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    /// Store the maximum value of \f$\log p(d|\vec\theta)\f$ encountered by the chain.  Used for determining when burn-in is complete.
    MPI_Send(&chain->logLmax, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
}


void mpi_recv_chain(struct Chain *chain, int source_rank)
{
    MPI_Status status;

    /// Number of chains
    MPI_Recv(&chain->NC, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    
    /// Array containing current order of chains in temperature ladder
    MPI_Recv(chain->index, chain->NC, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    
    /// Size of model being sampled by chain.  Depricated?
    //int **dimension;
    
    /// Array tracking acceptance rate of parallel chain exchanges.
    MPI_Recv(chain->acceptance, chain->NC, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    
    /// Array of chain temperatures
    MPI_Recv(chain->temperature, chain->NC, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    /// Array storing \f$\langle \log p(d|\vec\theta)^{1/T} \rangle\f$ for thermodynamic integration.
    MPI_Recv(chain->avgLogL, chain->NC, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    /// Store the maximum value of \f$\log p(d|\vec\theta)\f$ encountered by the chain.  Used for determining when burn-in is complete.
    MPI_Recv(&chain->logLmax, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
}
void mpi_send_all_info(int procID, struct UCBData *ucb_data, struct GlobalFitData *global_fit, int procID_target, int ucb_index, bool send_global_fit)
{
    for (int i = 0; i < ucb_data->chain->NC; i += 1)
    {
        mpi_send_model(ucb_data->model[i], procID_target);
        //printf("finish send source %d\n", i);
    }
    mpi_send_chain(ucb_data->chain, procID_target);
    mpi_send_data(ucb_data->data, procID_target);
    if (send_global_fit)
    {
        mpi_send_global_fit(global_fit, procID_target);
    }
}

// void mpi_process_send_out(int procID, struct Translator *translator, int procID_target, int ucb_index)
// {
//     MPI_Status status;
//     int indicator = 1;
//     int action_type = -1;
//     int ucb_index_receive = 0;
//     // int procID_tmp = 0; 
    
//     // adjust procID_target
//     MPI_Recv(&procID_target, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
//     MPI_Recv(&action_type, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD, &status);
//     printf("recv1 %d\n", action_type);
//     if (action_type == 2)
//     {
//         /// THIS SHOULD WORK, JUST NEED TO ADD INFORMATION ABOUT
//         /// UCB_INDEX GOING BACK AND FORTH, 
//         /// CLEARING OUT WHEN FREEING THE LAST PROCESSES.

//         MPI_Send(&indicator, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD);
//         MPI_Recv(&ucb_index_receive, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD, &status);
//         mpi_process_runner_inner(procID, translator->ucb_data_all[ucb_index_receive], translator->global_fit, procID_target, false);
//         printf("recv2 %d\n", action_type);
    
//         MPI_Recv(&procID_target, 1, MPI_INT, procID_target, 1, MPI_COMM_WORLD, &status);
        
//         MPI_Recv(&action_type, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD, &status);
//         printf("recv3 %d\n", action_type);
    
//         if (action_type != 1)
//         {
//             fprintf(stderr, "Action type not correct. \n");
//             exit(-1);
//         }
//     }
//     MPI_Send(&indicator, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD);
//     MPI_Send(&ucb_index, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD);
//     mpi_send_all_info(procID, translator->ucb_data_all[ucb_index], translator->global_fit, procID_target, ucb_index, true);  
// }



void mpi_process_runner_inner(int procID, struct UCBData *ucb_data, struct GlobalFitData *global_fit, int source_rank, bool send_global_fit)
{
    for (int i = 0; i < ucb_data->chain->NC; i += 1)
    {
        mpi_recv_model(ucb_data->model[i], source_rank);
        //printf("finish receive source %d\n", i);
    }
    mpi_recv_chain(ucb_data->chain, source_rank);
    mpi_recv_data(ucb_data->data, source_rank);
    if (send_global_fit)
    {
        mpi_recv_global_fit(global_fit, source_rank);
    }
    //printf("FInISH everything!!!! send\n");
    
}

// void mpi_process_runner(int procID, struct Translator *translator, int procID_min)
// {
//     bool run = true;
//     int ucb_index_now = 0;
//     int indicator = 0;
//     int send_indicator = 0;
//     MPI_Status status;
//     while (run)
//     {
//         printf("sending for info (%d)!\n", procID);
//         MPI_Send(&procID, 1, MPI_INT, procID_min, 1, MPI_COMM_WORLD);
//         send_indicator = 1;
//         MPI_Send(&send_indicator, 1, MPI_INT, procID_min, 0, MPI_COMM_WORLD);
//         printf("Waiting for mPI send (%d)!\n", procID);
//         MPI_Recv(&indicator, 1, MPI_INT, procID_min, 0, MPI_COMM_WORLD, &status);
//         printf("(%d) Received indicator: %d!\n", procID, indicator);
        
//         if (indicator == 1)
//         {
//             MPI_Recv(&ucb_index_now, 1, MPI_INT, procID_min, 0, MPI_COMM_WORLD, &status);
//             mpi_process_runner_inner(procID, translator->ucb_data, translator->global_fit, procID_min, true);
            
//             printf("STARTING UCB UPDATE %d %d \n", procID, ucb_index_now);
//             run_ucb_update(procID, translator->global_fit, translator->ucb_data, 1, 0, 0);
//             printf("ENDING UCB UPDATE %d %d \n", procID, ucb_index_now);
            
//             MPI_Send(&procID, 1, MPI_INT, procID_min, 1, MPI_COMM_WORLD);
//             printf("TMP2 %d %d \n", procID, ucb_index_now);
            
//             send_indicator = 2;
//             MPI_Send(&send_indicator, 1, MPI_INT, procID_min, 0, MPI_COMM_WORLD);
//             printf("TMP3 %d %d \n", procID, ucb_index_now);
            
//             MPI_Recv(&indicator, 1, MPI_INT, procID_min, 0, MPI_COMM_WORLD, &status);
//             printf("TMP4 %d %d \n", procID, ucb_index_now);
            
//             MPI_Send(&ucb_index_now, 1, MPI_INT, procID_min, 0, MPI_COMM_WORLD);
//             printf("TMP6 %d %d \n", procID, ucb_index_now);
            
//             mpi_send_all_info(procID, translator->ucb_data, translator->global_fit, procID_min, ucb_index_now, false);
//         }
//         else if (indicator == -1)
//         {
//             run = false;
//         }
//     }
//     printf("end indicator process %d\n", procID);
// }



void mpi_process_send_out(int procID, struct Translator *translator, int procID_min, int next_free_process, int ucb_index)
{
    MPI_Status status;
    int indicator = 1;
    int action_type = -1;
    int procID_target = procID_min + 1 + next_free_process; // next_free_process starts at 0
    // int procID_tmp = 0; 
    MPI_Send(&indicator, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD);
    MPI_Send(&ucb_index, 1, MPI_INT, procID_target, 0, MPI_COMM_WORLD);
    mpi_send_all_info(procID, translator->ucb_data_all[ucb_index], translator->global_fit, procID_target, ucb_index, true);  
    
    // update holder
    translator->ucb_index_running[next_free_process] = ucb_index;
}


void clear_mpi_process(int procID_min, struct Translator *translator)
{
    int indicator = -1;
    int procID_tmp = 0;
    int action_type = 0;
    int ucb_index_receive = 0;
    MPI_Status status;
    int ready = -1;
    MPI_Recv(&procID_tmp, 1, MPI_INT, MPI_ANY_SOURCE, 33, MPI_COMM_WORLD, &status);
    MPI_Recv(&ucb_index_receive, 1, MPI_INT, procID_tmp, 44, MPI_COMM_WORLD, &status);
    int process_index = procID_tmp - (procID_min + 1);  // process_index starts at 0
    int check_ucb_index = translator->ucb_index_running[process_index];
    if (check_ucb_index != ucb_index_receive)
    {
        fprintf(stderr, "Ucb indexes not matching, %d vs. %d.\n", check_ucb_index, ucb_index_receive);
        exit(-1);
    }
    mpi_process_runner_inner(procID_min, translator->ucb_data_all[ucb_index_receive], translator->global_fit, procID_tmp, false);
    translator->ucb_index_running[process_index] = -1;
}


void mpi_process_runner(int procID, struct Translator *translator, int procID_min)
{
    bool run = true;
    int ucb_index_now = 0;
    int indicator = 0;
    int send_indicator = 0;
    MPI_Status status;
    while (run)
    {    
        MPI_Recv(&indicator, 1, MPI_INT, procID_min, 0, MPI_COMM_WORLD, &status);
        
        if (indicator == 1)
        {
            MPI_Recv(&ucb_index_now, 1, MPI_INT, procID_min, 0, MPI_COMM_WORLD, &status);
            mpi_process_runner_inner(procID, translator->ucb_data, translator->global_fit, procID_min, true);
            run_ucb_update(procID, translator->global_fit, translator->ucb_data, 1, 0, 0);
            MPI_Send(&procID, 1, MPI_INT, procID_min, 33, MPI_COMM_WORLD);
            MPI_Send(&ucb_index_now, 1, MPI_INT, procID_min, 44, MPI_COMM_WORLD);
            mpi_send_all_info(procID, translator->ucb_data, translator->global_fit, procID_min, ucb_index_now, false);
        }
        else if (indicator == -1)
        {
            run = false;
        }
    }
    printf("end indicator process %d\n", procID);
}

void mpi_recv_global_fit(struct GlobalFitData *global_fit, int source_rank)
{
    MPI_Status status;
    MPI_Recv(&global_fit->nUCB, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&global_fit->nMBH, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&global_fit->nVGB, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);

    mpi_recv_tdi(global_fit->tdi_full, source_rank);
    mpi_recv_tdi(global_fit->tdi_store, source_rank);
    mpi_recv_tdi(global_fit->tdi_ucb, source_rank);
    mpi_recv_tdi(global_fit->tdi_vgb, source_rank);
    mpi_recv_tdi(global_fit->tdi_mbh, source_rank);
    mpi_recv_noise(global_fit->psd, source_rank);
    
    MPI_Recv(&global_fit->block_time, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&global_fit->max_block_time, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    // FILE *chainFile; // TODO: need?
};

void mpi_send_global_fit(struct GlobalFitData *global_fit, int receiver)
{
    MPI_Send(&global_fit->nUCB, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&global_fit->nMBH, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&global_fit->nVGB, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);

    mpi_send_tdi(global_fit->tdi_full, receiver);
    mpi_send_tdi(global_fit->tdi_store, receiver);
    mpi_send_tdi(global_fit->tdi_ucb, receiver);
    mpi_send_tdi(global_fit->tdi_vgb, receiver);
    mpi_send_tdi(global_fit->tdi_mbh, receiver);
    mpi_send_noise(global_fit->psd, receiver);
    
    MPI_Send(&global_fit->block_time, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&global_fit->max_block_time, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    // FILE *chainFile; // TODO: need?
};

void mpi_send_data(struct Data *data, int receiver)
{
    MPI_Send(&data->N, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->Nchannel, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->logN, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    
    /**  Random Number Generator Seeds */
    MPI_Send(&data->cseed, 1, MPI_LONG, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->nseed, 1, MPI_LONG, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->iseed, 1, MPI_LONG, receiver, 0, MPI_COMM_WORLD);

    /** @name Time of Observations */
    MPI_Send(&data->T, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->sqT, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->t0, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    /** @name Analysis Frequency Segment */
    MPI_Send(&data->qmin, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->qmax, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->qpad, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    
    MPI_Send(&data->fmin, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->fmax, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->sine_f_on_fstar, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    //some manipulations of f,fmin for likelihood calculation
    MPI_Send(&data->sum_log_f, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->logfmin, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->logfmax, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    /** @name TDI Data and Noise */
    mpi_send_tdi(data->tdi, receiver);
    mpi_send_tdi(data->raw, receiver);
    mpi_send_noise(data->noise, receiver);
    
    // //Spectrum proposal
    // double *p; //!<power spectral density of data
    MPI_Send(&data->pmax, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&data->SNR2, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
}

void mpi_recv_data(struct Data *data, int source_rank)
{
    MPI_Status status;

    MPI_Recv(&data->N, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->Nchannel, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->logN, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    
    /**  Random Number Generator Seeds */
    MPI_Recv(&data->cseed, 1, MPI_LONG, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->nseed, 1, MPI_LONG, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->iseed, 1, MPI_LONG, source_rank, 0, MPI_COMM_WORLD, &status);

    /** @name Time of Observations */
    MPI_Recv(&data->T, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->sqT, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->t0, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    /** @name Analysis Frequency Segment */
    MPI_Recv(&data->qmin, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->qmax, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->qpad, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    
    MPI_Recv(&data->fmin, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->fmax, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->sine_f_on_fstar, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    //some manipulations of f,fmin for likelihood calculation
    MPI_Recv(&data->sum_log_f, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->logfmin, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->logfmax, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);

    /** @name TDI Data and Noise */
    mpi_recv_tdi(data->tdi, source_rank, &status);
    mpi_recv_tdi(data->raw, source_rank, &status);
    mpi_recv_noise(data->noise, source_rank, &status);
    
    // //Spectrum proposal
    // double *p; //!<power spectral density of data
    MPI_Recv(&data->pmax, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&data->SNR2, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
}


void mpi_recv_source(struct Source *origin, int source_rank)
{
    MPI_Status status;

    //Intrinsic
    
    MPI_Recv(&origin->m1, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->m2, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->f0, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
        
    //Extrinisic
    MPI_Recv(&origin->psi, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->cosi, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->phi0, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    MPI_Recv(&origin->D, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->phi, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->costheta, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);

    //Derived
    MPI_Recv(&origin->amp, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->Mc, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->dfdt, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->d2fdt2, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    //Book-keeping
    MPI_Recv(&origin->BW, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->qmin, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->qmax, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->imin, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->imax, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    
    // //Response
    MPI_Recv(&origin->tdi->N, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->tdi->Nchannel, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    
    // printf("MPI1: %d %d %d %d, %d %e\n", n, model->Nmax, model->Neff, model->Nlive, origin->imax, origin->tdi->A[1]);

    MPI_Recv(origin->tdi->A, 2*origin->tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(origin->tdi->E, 2*origin->tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(origin->tdi->T, 2*origin->tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(origin->tdi->X, 2*origin->tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(origin->tdi->Y, 2*origin->tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(origin->tdi->Z, 2*origin->tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    

    // //Fisher
    MPI_Recv(origin->fisher_evalue, UCB_MODEL_NP, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(origin->params, UCB_MODEL_NP, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&origin->fisher_update_flag, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);

    for(int i=0; i<UCB_MODEL_NP; i++)
    {
        MPI_Recv(origin->fisher_matrix[i], UCB_MODEL_NP, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(origin->fisher_evectr[i], UCB_MODEL_NP, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    }
    // printf("MPI2: %d %d %d %d, %d %e %e\n", n, model->Nmax, model->Neff, model->Nlive, origin->imax, origin->tdi->A[1], origin->fisher_evectr[0][0]);

}


void mpi_send_source(struct Source *origin, int receiver)
{
    MPI_Status status;

    //Intrinsic
    MPI_Send(&origin->m1, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->m2, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->f0, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
        
    //Extrinisic
    MPI_Send(&origin->psi, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->cosi, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->phi0, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    MPI_Send(&origin->D, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->phi, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->costheta, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    // Derived
    MPI_Send(&origin->amp, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->Mc, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->dfdt, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->d2fdt2, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    //Book-keeping
    MPI_Send(&origin->BW, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->qmin, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->qmax, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->imin, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->imax, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    
    // Response
    MPI_Send(&origin->tdi->N, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->tdi->Nchannel, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    
    MPI_Send(origin->tdi->A, 2*origin->tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(origin->tdi->E, 2*origin->tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(origin->tdi->T, 2*origin->tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(origin->tdi->X, 2*origin->tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(origin->tdi->Y, 2*origin->tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(origin->tdi->Z, 2*origin->tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    // INFO MAT
    MPI_Send(origin->fisher_evalue, UCB_MODEL_NP, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(origin->params, UCB_MODEL_NP, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&origin->fisher_update_flag, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);

    for(int i=0; i<UCB_MODEL_NP; i++)
    {
        MPI_Send(origin->fisher_matrix[i], UCB_MODEL_NP, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
        MPI_Send(origin->fisher_evectr[i], UCB_MODEL_NP, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    }
    // printf("model source %d %d %e\n", n, origin->imax, origin->tdi->A[1]);

}

void mpi_recv_noise(struct Noise *noise, int source_rank)
{
    MPI_Status status;

    //Noise parameters
    MPI_Recv(&noise->N, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&noise->Nchannel, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    
    MPI_Recv(noise->eta, noise->Nchannel, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(noise->f, noise->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);

    // TODO: need to fix this for more channels
    for(int i=0; i<noise->Nchannel; i++)
    {
        // for(int j=0; j<3; j++)
        // {
        int j = i;
            for (int k = 0; k < noise->N; k++)
            {
                double tmp1, tmp2;
                
                MPI_Recv(&noise->C[i][j][k], 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&noise->invC[i][j][k], 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
                
                // noise->C[i][j][k] = tmp1;
                // noise->invC[i][j][k] = tmp2;
                // printf("receiv: %d %d %d %e %e\n", i, j, k, noise->C[i][j][k], noise->invC[i][j][k]);
                
            }
        // }
    }
    
    MPI_Recv(noise->detC, noise->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(noise->transfer, noise->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
}


void mpi_send_noise(struct Noise *noise, int receiver)
{
    MPI_Status status;

    //Noise parameters
    MPI_Send(&noise->N, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&noise->Nchannel, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    
    MPI_Send(noise->eta, noise->Nchannel, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(noise->f, noise->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);

    for(int i=0; i<noise->Nchannel; i++)
    {
        // for(int j=0; j<3; j++)
        // {
        int j = i;
            for (int k = 0; k < noise->N; k++)
            {
                double tmp1, tmp2;
                // tmp1 = noise->C[i][j][k];
                // tmp2 = noise->invC[i][j][k];
                // printf("%d %d %d %e %e\n", i, j, k, noise->C[i][j][k], noise->invC[i][j][k]);
                MPI_Send(&noise->C[i][j][k], 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
                MPI_Send(&noise->invC[i][j][k], 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
            }        
        // }
    }
    
    MPI_Send(noise->detC, noise->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(noise->transfer, noise->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
}

void mpi_recv_tdi(struct TDI *tdi, int source_rank)
{
    MPI_Status status;

    MPI_Recv(&tdi->N, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&tdi->Nchannel, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    
    // TDI
    MPI_Recv(tdi->A, 2*tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(tdi->E, 2*tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(tdi->T, 2*tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(tdi->X, 2*tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(tdi->Y, 2*tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(tdi->Z, 2*tdi->N, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
}


void mpi_send_tdi(struct TDI *tdi, int receiver)
{
      // //TDI
    // Response
    MPI_Send(&tdi->N, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&tdi->Nchannel, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    
    MPI_Send(tdi->A, 2*tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(tdi->E, 2*tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(tdi->T, 2*tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(tdi->X, 2*tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(tdi->Y, 2*tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(tdi->Z, 2*tdi->N, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
}


void mpi_recv_model(struct Model *model, int source_rank)
{
    MPI_Status status;
    //Source parameters
    MPI_Recv(&model->Nmax, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&model->Neff, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&model->Nlive, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);

    // SOURCES
    for(int n=0; n<model->Nlive; n++)
    {
        mpi_recv_source(model->source[n], source_rank);
    }

    mpi_recv_noise(model->noise, source_rank);

    mpi_recv_tdi(model->tdi, source_rank);
    mpi_recv_tdi(model->residual, source_rank);

    //Start time for segment for model
    MPI_Recv(&model->t0, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
    //Source parameter priors
    for(int n=0; n<UCB_MODEL_NP; n++)
    {
        for(int j=0; j<2; j++) 
        {
            MPI_Recv(&model->prior[n][j], 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
        }
        MPI_Recv(&model->logPriorVolume[n], 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    }
    MPI_Recv(&model->logL, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&model->logLnorm, 1, MPI_DOUBLE, source_rank, 0, MPI_COMM_WORLD, &status);
    
}

void mpi_send_model(struct Model *model, int receiver)
{
    // Source parameters
    MPI_Send(&model->Nmax, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&model->Neff, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&model->Nlive, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);

    // SOURCES
    for(int n=0; n<model->Nlive; n++)
    {
        mpi_send_source(model->source[n], receiver);
    }
    //
    
    //Noise parameters
    mpi_send_noise(model->noise, receiver);

     // //Calibration parameters
    // copy_calibration(model->calibration,copy->calibration);
    
    // TDI
    mpi_send_tdi(model->tdi, receiver);
    mpi_send_tdi(model->residual, receiver);
    
    //Start time for segment for model
    MPI_Send(&model->t0, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    
    //Source parameter priors
    for(int n=0; n<UCB_MODEL_NP; n++)
    {
        for(int j=0; j<2; j++) 
        {
            MPI_Send(&model->prior[n][j], 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
        }
        MPI_Send(&model->logPriorVolume[n], 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    }
    //Model likelihood
    MPI_Send(&model->logL, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(&model->logLnorm, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
}
