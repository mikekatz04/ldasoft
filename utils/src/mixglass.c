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
    
    sprintf(ucb_data->data->dataDir,"%sdata",flags->runDir);
    sprintf(ucb_data->chain->chainDir,"%schains",flags->runDir);
    sprintf(ucb_data->chain->chkptDir,"%scheckpoint",flags->runDir);
    
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

void get_current_glass_params(struct Translator *translator, double *params, int *nleaves, double *logl, double *logp, double* betas, int array_length)
{
    struct UCBData *ucb_data = translator->ucb_data;
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
    

void set_current_glass_params(struct Translator *translator, double *params, int *nleaves, double *logl, double *logp, double *betas, int array_length)
{
    struct UCBData *ucb_data = translator->ucb_data;
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
    
    struct Chain *chain = ucb_data->chain;
    struct Data *data = ucb_data->data;
    
    int N2 = data->N*2;
    if (N != data->N)
    {
        fprintf(stderr,"Input and output lengths are different. N in: %d ; N in c backend: %d\n", N, data->N);
        exit(1);
    }

    if (Nchannel != data->Nchannel)
    {
        fprintf(stderr,"Input and output number of channels are different. Nchannel in: %d ; Nchannel in c backend: %d\n", Nchannel, data->Nchannel);
        exit(1);
    }

    for(int i=0; i<N2; i++)
    {
        //printf("OEWFEOWJ: %e %e\n", model->noise->C[0][0][0], model->noise->C[1][1][0]);
                
        switch(data->Nchannel)
        {
            case 2:
                data_arr[0 * N2 + i] = data->tdi->A[i] - model->tdi->A[i];
                data_arr[1 * N2 + i] = data->tdi->E[i] - model->tdi->E[i];
                break;
            case 3:
                data_arr[0 * N2 + i] = data->tdi->X[i] - model->tdi->X[i];
                data_arr[1 * N2 + i] = data->tdi->Y[i] - model->tdi->Y[i];
                data_arr[2 * N2 + i] = data->tdi->Z[i] - model->tdi->Z[i];
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

void get_main_tdi_data_in_glass(struct Translator *translator, double *data_arr, int Nchannel, int N)
{
    struct UCBData *ucb_data = translator->ucb_data;
    
    struct Chain *chain = ucb_data->chain;
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
    struct UCBData *ucb_data = translator->ucb_data;

    struct Chain *chain = ucb_data->chain;
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
    struct UCBData *ucb_data = translator->ucb_data;
    
    struct Chain *chain = ucb_data->chain;
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
    struct UCBData *ucb_data = translator->ucb_data;
    
    struct Chain *chain = ucb_data->chain;
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
