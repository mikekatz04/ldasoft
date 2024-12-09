#ifndef __UCB_MCMC_BASE_H__
#define __UCB_MCMC_BASE_H__

#include "glass_data.h"
#include "glass_ucb_waveform.h"
#include "glass_utils.h"
#include "glass_lisa.h"

// void run_ucb_mcmc(struct Data *data, struct Orbit *orbit, struct Flags *flags, struct Chain *chain, struct Source *inj);
static void print_usage();
void run_ucb_mcmc(struct Data *data, struct Orbit *orbit, struct Flags *flags, struct Chain *chain, struct Source *inj);

#endif // __UCB_MCMC_BASE_H__