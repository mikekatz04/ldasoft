#include "mixglass.h"
#include "stdio.h"
#include "stdlib.h"
#include "glass_data.h"
#include "ucb_mcmc_base.h"

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
    flags->no_vgb = translator->no_vgb;
    flags->no_noise = translator->no_noise;
    
    /*
     default data format is 'phase'
     optional support for 'frequency' a la LDCs
     */
    memcpy(&data->format, &translator->format, 16 * sizeof(char));
        
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
