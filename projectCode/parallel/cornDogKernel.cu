/* file:	cornDogKernel.cu
 * desc:	Corn dog stand simulation kernel functions.
 *        Modified from the sequential version in cornDogModule.c
 * date:	12/15/18
 * name:	Chris Saiko */

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "support.h"       //headers and structs

// curand_setup_kernel function
// sets up an array of states that each thread uses to generate a random number
// from stackoverflow.com, Robert Crovella
__global__ void curand_setup_kernel(curandState *state, int seed){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
  __syncthreads();
}

// CDtest_Kernel function
// Tests argument passing to the GPU
__global__ void CDtest_Kernel(int sims, struct simArgs* args, curandState* d_states, int* simScores){

  // thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  simScores[idx] = idx + 1;

}//end CDtest_Kernel


// cornDogKernel function - child kernel
// Runs a single corndog simulation on the GPU, saving the result
// to a location in shared memory
__global__ void cornDog_Kernel(int sims, struct simArgs* args, curandState* d_states, int* simScores) {

  int value = 0;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // get the d_state element for this thread
  curandState state = d_states[idx];

  // declare and allocate some structs for this thread's simulation
  struct stand* stand = (struct stand*)malloc(sizeof(struct stand));
  struct behavior* behavior = (struct behavior*)malloc(sizeof(struct behavior));

  // clear the memory to zero
  memset(stand, 0, sizeof(struct stand));
  memset(behavior, 0, sizeof(struct behavior));

  // bounds check if it is a thread past a simulation number we don't want
  if (idx < sims){

    // extract arguments to variables
    // corndog stand variables
    stand->numFryers = args->numFryers;
    stand->maxFryerSize = args->maxFryerSize;
    stand->numCustQueues = args->numCustQueues;
    stand->openHour = args->openHour;
    stand->closeHour = args->closeHour;
    stand->corndogPrice = args->corndogPrice;
    stand->corndogCost = args->corndogCost;
    stand->corndogCookTime = args->corndogCookTime;
    stand->wage = args->wage;
    stand->customerPerMin = args->customerPerMin;

    // customer behaviors
    behavior->drinkChance = args->drinkChance;
    behavior->drinkTempChance = args->drinkTempChance;
    behavior->foodChance = args->foodChance;
    behavior->foodMealChance = args->foodMealChance;
    behavior->customerMaxQueueTime = args->customerMaxQueueTime;
    behavior->customerChance = args->customerChance;
    behavior->customerChanceHour = args->customerChanceHour;

    // get hourly temps
    int t;
    for (t = 0; t < 24; t++){
      stand->hourlyTemps[t] = args->hourlyTemps[t];
    }

    // set up other sim variables
  	stand->cornDogsWasted = 0;
  	stand->lostCustomers = 0;
  	stand->waitingCustomers = 0;
  	stand->servedCustomers = 0;
  	stand->foodItems = 0;
  	stand->drinkSales = 0;
  	stand->corndogSales = 0;
  	stand->profit = 0;
  	stand->totalWages = 0;
  	stand->walkingPast = 0;
  	stand->currentHour = stand->openHour;
  	stand->dailyScore = 0;

  	stand->drinkPrice = 2.0;
  	stand->drinkCost = 0.75;

    // set up fryer buffers to contain cooking corn dogs
  	struct cornDog* fryer1;
    struct cornDog* fryer2;
  	if (stand->numFryers >= 1) {
  		fryer1 = (struct cornDog*)malloc(sizeof(struct cornDog));
  		fryer1->next = NULL;
      fryer1->prev = NULL;
  		fryer1->age = -1;	//head of list
      fryer1->items = 0; //empty list
      fryer1->max = stand->maxFryerSize;
  	}
  	if (stand->numFryers == 2) {
  		fryer2 = (struct cornDog*)malloc(sizeof(struct cornDog));
  		fryer2->next = NULL;
      fryer2->prev = NULL;
  		fryer2->age = -1;	//head of list
      fryer2->items = 0; //empty list
      fryer2->max = stand->maxFryerSize;
  	}

    // set up customer queue buffers to contain waiting customers
  	struct customer* custLine1;
  	struct customer* custLine2;
  	if (stand->numCustQueues >= 1)	{
  		custLine1 = (struct customer*)malloc(sizeof(struct customer));
  		custLine1->waitTime = -1; //head of list
      custLine1->next = NULL;
      custLine1->length = 0; //empty list
  	}
  	if (stand->numCustQueues == 2) {
  		custLine2 = (struct customer*)malloc(sizeof(struct customer));
  		custLine2->waitTime = -1; //head of list
      custLine2->next = NULL;
      custLine2->length = 0; //empty list
  	}

  	// set up the food buffer, containing cooked corn dogs
  	struct cornDog* cookedFood;
  	cookedFood = (struct cornDog*)malloc(sizeof(struct cornDog));
  	cookedFood->next = NULL;
    cookedFood->prev = NULL;
  	cookedFood->age = -1;	//head of list
    cookedFood->items = 0;	//empty list

    // get the number of ticks (minutes) that the simulation will run over
  	int i;
  	int ticks = (stand->closeHour - stand->openHour) * 60;

    // simulate the corn dog stand
  	for (i = 0; i < ticks; i++){

  		// update variables
  		stand->currentHour = stand->openHour + (i / 60);

  		// food buffer handler
      foods_handler(cookedFood, stand);

      // for each fryer
  		// fryer buffer handler
      if (stand->numFryers >= 1) {
        fryer_handler(fryer1, cookedFood, stand);
      }
      if (stand->numFryers == 2) {
        fryer_handler(fryer2, cookedFood, stand);
      }

  		// customer line buffer handler
      if (stand->numCustQueues == 1)	{
        custLine_handler(custLine1, cookedFood, stand, behavior);
    	}
    	if (stand->numCustQueues == 2) {
        custLine_handler(custLine1, cookedFood, stand, behavior);
        custLine_handler(custLine2, cookedFood, stand, behavior);
    	}

  		// new customers handler
      if (stand->numCustQueues == 1)	{
        newCust_handler(custLine1, stand, behavior, state);
    	}
    	if (stand->numCustQueues == 2) {
        if (custLine1->length < custLine2->length) newCust_handler(custLine1, stand, behavior, state);
        else newCust_handler(custLine2, stand, behavior, state);
    	}

  	}//end corn dog sim for loop

    // stand cleanup
  	stand->cornDogsWasted += cookedFood->items;

  	// for each fryer
  	if (stand->numFryers >= 1) {
  		stand->cornDogsWasted += fryer1->items;
  	}
  	if (stand->numFryers == 2) {
  		stand->cornDogsWasted += fryer2->items;
  	}

    // consider waiting customers at the end of the day as lost
  	stand->lostCustomers += stand->waitingCustomers;

  	// total wages paid
  	stand->totalWages = ((stand->closeHour - stand->openHour) * stand->wage * (stand->numFryers + stand->numCustQueues));

  	// total sales for the day
  	stand->profit = (stand->corndogSales * (stand->corndogPrice - stand->corndogCost) - stand->cornDogsWasted * stand->corndogCost);
  	stand->profit = (stand->profit + stand->drinkSales * (stand->drinkPrice - stand->drinkCost));
  	stand->profit -= stand->totalWages;

    // calculate daily score, higher score is better
  	stand->dailyScore = (1 + stand->servedCustomers * 3);
  	stand->dailyScore = (stand->dailyScore - (stand->lostCustomers * 2) - (stand->cornDogsWasted * 1));
  	stand->dailyScore += (4 * stand->profit);
    value = stand->dailyScore;

    // store score into memory
    simScores[idx] = value;

    //free memory
    //free(fryer1);
    //free(fryer2);
    //free(custLine1);
    //free(custLine2);
    //free(cookedFood);

  }//end bounds check if

  // free allocated memory for the thread
  free(stand);
  free(behavior);

  __syncthreads();

}//end cornDog_Kernel function


// sim_ann_kernel function - parent kernel
// Runs the simulated annealing algorithm, calling child threads
__global__ void sim_ann_kernel(int sims, struct simArgs* args, int* simScores){

  int val = 0;
  int i;
	int bestValue = 0;
  long long int time_val = clock64();
  float elapsed = 0;

  float alpha = 0.75;					//def 0.98
  float temperature = 1000.0;	//def 1000.0
  float end = 0.01;						//def 0.01
  float proba;
  int iterations = 50000;			//def 50000
  int iters = 0;
  int delta;

  //save values for output later
  float output_alpha = alpha;
  float output_temperature = temperature;
  float output_end = end;
  int output_iterations = iterations;

  // best args initialized
  struct simArgs* bestArgs = (struct simArgs*) malloc(sizeof(struct simArgs));
  best_args(args, bestArgs);

  // setup for child kernels
  // grid setup for the number of sims/threads
  int blockSize = 1024;
  int numBlocks = sims/blockSize;
  if (sims % blockSize != 0) numBlocks++;

  // allocate device memory for the simulation scores
  int * d_simScores;
  cudaMalloc(&d_simScores, sims * (sizeof(int)));
  //cudaMemset(d_simScores, 0, sims * sizeof(int));

  // allocate device memory for the arguments
  struct simArgs* d_args;
  cudaMalloc(&d_args, (sizeof(struct simArgs)));
  //cudaMemset(d_args, 0, sizeof(struct simArgs));

  // initialize random numbers on GPU for these threads
  // from stackoverflow.com, Robert Crovella
  curandState *d_states;
  cudaMalloc(&d_states, sims * (sizeof(curandState)));
  // srand(time(0));
  // int seed = rand();  //set random number for seed
  int seed = 5351;

  // while(elapsed < seconds){
  //
  //
  //
  //
  //
  //   //printf("SIM VALUE %d\n",value);
  //   if (val > bestValue){
  //     //save new best value
  //     bestValue = val;
  //
  //     best_args(args, bestArgs);
  //     printf("New best args: %d Elapsed: %f Iters: %d\n",val,elapsed,iters);
  //
  //   }//end if
  //   else{
  //     best_args(bestArgs, args);	//go back to other args
  //     find_neighbor_d(args, d_states[0]);			//find new neighbor
  //   }
  //
  //   if (clock64() > time_val){
  //     elapsed += ((clock64() - time_val)/1733000000.0);
  //     time_val = clock64();
  //   }
  //
  // }//end while

	bestValue = val;	//equal at this point

	while(temperature > end){
		iters++;

		find_neighbor_d(args, d_states[0]);			//find new neighbor
		printf("Temperature: %f\n",temperature);
		// run simulation to get score value
		val = 0;

    // set up random kernel state for each thread
    // used when generating random numbers in each thread
    curand_setup_kernel<<<numBlocks,blockSize>>>(d_states, seed);

    // sim with these arguments
    // simlation child kernel call
    cornDog_Kernel<<<numBlocks,blockSize>>>(sims, args, d_states, d_simScores);

    // sequential get value
    for (i = 0; i < sims; i++){
      val += d_simScores[i];
    }
    val /= sims;


		delta = bestValue - val;
		if (delta < 0){
			//better score
			bestValue = val;
			best_args(args,bestArgs);
			printf("New best args: \n");
		}
		else{
			proba = curand_uniform(&d_states[0]);

			if (proba < expf(-delta/temperature) ){
				//keep worse value
				bestValue = val;
				best_args(args,bestArgs);
				printf("New worse best args: \n");
			}//end probability if
		}//end else

		// cooling process for temperature
		temperature *= alpha;

	}//end temperature while

  printf("Temperature cooled!\n");
  //copy best args back to args, to return
  best_args(bestArgs, args);
  //copy best value, to return
  simScores[0] = bestValue;
  printf("Best Value to return: %d\n",simScores[0]);
  printf("Best Args to return\n");
  print_args_d(args);
  return;


}//end sim_ann_kernel function


// hill_climbing_kernel function - parent kernel
// Runs the hill climbing algorithm, calling child threads
__global__ void hill_climbing_kernel(int sims, int seconds, struct simArgs* args, int* simScores) {

  int val = 0;
  int i,s;
  int bestValue = 0;
  long long int time_val = clock64();
  float elapsed = 0;
  int iters = 0;

  // best args initialized
  struct simArgs* bestArgs = (struct simArgs*) malloc(sizeof(struct simArgs));
  best_args(args, bestArgs);

  // setup for child kernels
  // grid setup for the number of sims/threads
  int blockSize = 1024;
  int numBlocks = sims/blockSize;
  if (sims % blockSize != 0) numBlocks++;

  // allocate device memory for the simulation scores
  int * d_simScores;
  cudaMalloc(&d_simScores, sims * (sizeof(int)));
  //cudaMemset(d_simScores, 0, sims * sizeof(int));

  // allocate device memory for the arguments
  struct simArgs* d_args;
  cudaMalloc(&d_args, (sizeof(struct simArgs)));
  //cudaMemset(d_args, 0, sizeof(struct simArgs));

  // initialize random numbers on GPU for these threads
  // from stackoverflow.com, Robert Crovella
  curandState *d_states;
  cudaMalloc(&d_states, sims * (sizeof(curandState)));
  
  curandState *n_states;
  cudaMalloc(&n_states, (sizeof(curandState)));

  // int seed = rand();  //set random number for seed
  int seed = 5351;

  //set up random kernel state for each thread
  //used when generating random numbers in each thread
  curand_setup_kernel<<<numBlocks,blockSize>>>(d_states, seed);
  //curand_setup_kernel<<<1,1>>>(n_states, seed+1);
  //curandState nstate = d_states[0];

  while(elapsed < seconds){

    val = 0;	//reset value
    iters++;
    s = iters % 10;
    //if ((iters % 5091) == 0) printf("Iters: %d, s: %d\n",iters, s);
    //if ((iters % 91) == 0) printf("Iters: %d, s: %d\n",iters, s);

    // sim with these arguments
    // simlation child kernel call
    cornDog_Kernel<<<numBlocks,blockSize>>>(sims, args, d_states, d_simScores);

    // sequential get value
    for (i = 0; i < sims; i++){
      val += d_simScores[i];
    }
    val /= sims;

    //printf("SIM VALUE %d\n",value);
    if (val > bestValue){
      //save new best value
      bestValue = val;

      best_args(args, bestArgs);
      find_neighbor_d(args, d_states[0]); //new neighbor

      //printf("New best args: %d Elapsed: %f Iters: %d\n",val,elapsed,iters);

    }//end if
    else{
      best_args(bestArgs, args);	//go back to other args
      find_neighbor_d(args, d_states[0]);			//find new neighbor
    }

    if (clock64() > time_val){
      elapsed += ((clock64() - time_val)/1733000000.0);
      time_val = clock64();
    }

  }//end while

  printf("Time elapsed!\n");
  printf("Iterations: %d\n",iters);
  //copy best args back to args, to return
  best_args(bestArgs, args);
  //copy best value, to return
  simScores[0] = bestValue;
  printf("Best Value to return: %d\n",simScores[0]);
  printf("Best Args to return\n");
  print_args_d(args);
  printf("Memory error here somewhere, hangs for a moment..");
  return;

}//end hill_climbing_kernel



// tree_search_kernel function - parent kernel
// Runs the tree search algorithm, calling child threads
__global__ void tree_search_kernel(int sims, int seconds, struct simArgs* args, int* simScores) {

  int val = 0;
  int i;
  int bestValue = 0;
  long long int time_val = clock64();
  float elapsed = 0;
  int iters = 0;

  // best args initialized
	struct simArgs* bestArgs = (struct simArgs*) malloc(sizeof(struct simArgs));
  best_args(args, bestArgs);

  // setup for child kernels
  // grid setup for the number of sims/threads
  int blockSize = 1024;
  int numBlocks = sims/blockSize;
  if (sims % blockSize != 0) numBlocks++;

  // allocate device memory for the simulation scores
  int * d_simScores;
  cudaMalloc(&d_simScores, sims * (sizeof(int)));
  //cudaMemset(d_simScores, 0, sims * sizeof(int));

  // allocate device memory for the arguments
  struct simArgs* d_args;
  cudaMalloc(&d_args, (sizeof(struct simArgs)));
  //cudaMemset(d_args, 0, sizeof(struct simArgs));

  // initialize random numbers on GPU for these threads
  // from stackoverflow.com, Robert Crovella
  curandState *d_states;
  cudaMalloc(&d_states, sims * (sizeof(curandState)));
  // srand(time(0));
  // int seed = rand();  //set random number for seed
  int seed = 5351;

	//save initial arguments to local variables to use in for loops
	// corndog stand variables
  int numFryers = args->numFryers;
	int maxFryerSize = args->maxFryerSize;
	int numCustQueues = args->numCustQueues;
  int openHour = args->openHour;
  int closeHour = args->closeHour;
  float corndogPrice = args->corndogPrice;
  float corndogCost = args->corndogCost;
  int corndogCookTime = args->corndogCookTime;
  float wage = args->wage;

  // customer behaviors
  float drinkChance = args->drinkChance;
  float drinkTempChance = args->drinkTempChance;
  float foodChance = args->foodChance;
  float foodMealChance = args->foodMealChance;
  int customerMaxQueueTime = args->customerMaxQueueTime;
  float customerChance = args->customerChance;
  float customerChanceHour = args->customerChanceHour;

  //set up random kernel state for each thread
  //used when generating random numbers in each thread
  curand_setup_kernel<<<numBlocks,blockSize>>>(d_states, seed);

	// many for loops, the order of these will have a huge difference
	for(args->numFryers = numFryers; args->numFryers < 3; args->numFryers += 1){
		for(args->maxFryerSize = maxFryerSize; args->maxFryerSize < 7; args->maxFryerSize += 1){
			for(args->numCustQueues = numCustQueues; args->numCustQueues < 3; args->numCustQueues += 1){
				for(args->openHour = openHour; args->openHour < 8; args->openHour += 1){
					for(args->closeHour = closeHour; args->closeHour < 24; args->closeHour += 1){
						for(args->corndogPrice = corndogPrice; args->corndogPrice < 4.1; args->corndogPrice += 0.25){
							for(args->corndogCost = corndogCost; args->corndogCost < 2.1; args->corndogCost += 0.25){
								for(args->corndogCookTime = corndogCookTime; args->corndogCookTime < 5; args->corndogCookTime += 1){
									for(args->wage = wage; args->wage < 9.1; args->wage += 0.25){
										for(args->drinkChance = drinkChance; args->drinkChance < 0.81; args->drinkChance += 0.1){
											for(args->drinkTempChance = drinkTempChance; args->drinkTempChance < 0.201; args->drinkTempChance += 0.05){
												for(args->foodChance = foodChance; args->foodChance < 0.95; args->foodChance += 0.1){
													for(args->foodMealChance = foodMealChance; args->foodMealChance < 0.201; args->foodMealChance += 0.05){
														for(args->customerMaxQueueTime = customerMaxQueueTime; args->customerMaxQueueTime < 8; args->customerMaxQueueTime += 1){
															for(args->customerChance = customerChance; args->customerChance < 0.801; args->customerChance += 0.05){
																for(args->customerChanceHour = customerChanceHour; args->customerChanceHour < 0.151; args->customerChanceHour += 0.05){

																			//simulation
																			// reset value
																			val = 0;
                                      iters++;

																			// sim with these arguments
                                      // simlation child kernel call
                                      cornDog_Kernel<<<numBlocks,blockSize>>>(sims, args, d_states, d_simScores);

                                      // sequential get value
                                      for (i = 0; i < sims; i++){
                                        val += d_simScores[i];
                                      }
                                      val /= sims;

																			//printf("SIM VALUE %d\n",value);
																			if (val > bestValue){
																				//save new best value
																				bestValue = val;

																				best_args(args, bestArgs);
																				//printf("New best args: %d Elapsed: %f Iters: %d\n",val,elapsed,iters);

																			}//end if
                                      //update time

                                      if (clock64() > time_val){
                                        elapsed += ((clock64() - time_val)/1733000000.0);
                                        time_val = clock64();
                                      }
																			if (elapsed > seconds){ //time limit reached
                                        printf("\nTime elapsed!\n");
																				//copy best args back to args, to return
																				best_args(bestArgs, args);
                                        //copy best value, to return
                                        simScores[0] = bestValue;
					printf("Iterations: %d\n", iters);
                                        printf("Best Value to return: %d\n",simScores[0]);
                                        printf("Best Args to return\n");
                                        print_args_d(args);
                                        printf("Memory error here somewhere, it hangs a few moments...");
					return;
																			}//end if

																}//end customerChanceHour for
															}//end customerChance for
														}//end customerMaxQueueTime for
													}//end foodMealChance for
												}//end foodChance for
											}//end drinkTempChance for
										}//end drinkChance for
									}//end wage for
								}//end corndogCookTime for
							}//end corndogCost for
						}//end corndogPrice for
					}//end closeHour for
				}//end openHour for
			}//end numCustQueues for
		}//end maxFryerSize for
	}//end numFryers for

  printf("Iters: %d\n", iters);
  //copy best args back to args, to return
  best_args(bestArgs, args);
  //copy best value, to return
  simScores[0] = bestValue;
  printf("Best args: %d\n",simScores[0]);
  return;

}//end tree_search_kernel



// cornDogKernel function
// Runs "sims" number of corndog simulations on the GPU, then averages the result
int cornDogSim(int sims, struct simArgs* args) {

  // grid setup for the number of sims/threads
  int blockSize = 1024;
  int numBlocks = sims/blockSize;
  if (sims % blockSize != 0) numBlocks++;

  // allocate host memory for the simulation scores
  int * h_simScores = (int*)malloc(sizeof(int) * sims);

  // allocate device memory for the simulation scores
  int * d_simScores;
  cudaMalloc(&d_simScores, sims * (sizeof(int)));
  cudaMemset(d_simScores, 0, sims * sizeof(int));

  // allocate device memory for the arguments
  struct simArgs* d_args;
  cudaMalloc(&d_args, (sizeof(struct simArgs)));
  cudaMemset(d_args, 0, sizeof(struct simArgs));

  // copy arguments to device
  cudaMemcpy(d_args, args, sizeof(struct simArgs), cudaMemcpyHostToDevice);

  // initialize random numbers on GPU for these threads
  // from stackoverflow.com, Robert Crovella
  curandState *d_states;
  cudaMalloc(&d_states, sims * (sizeof(curandState)));
  srand(time(0));
  int seed = rand();  //set random number for seed

  // set heap size to 128 MB for in kernel malloc calls
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

  // set up random kernel state for each thread
  // used when generating random numbers in each thread
  curand_setup_kernel<<<numBlocks,blockSize>>>(d_states, seed);

  // simlation kernel call
  cornDog_Kernel<<<numBlocks,blockSize>>>(sims, d_args, d_states, d_simScores);

  //cudaDeviceSynchronize();

  // parallel sum the simulation results


  // copy scores back to host
  cudaMemcpy(h_simScores, d_simScores, sims * sizeof(int), cudaMemcpyDeviceToHost);

  // sequential code, get sum
  int i;
  int value = 0;
  for (i = 0; i < sims; i++){
    value += h_simScores[i];
  }
  printf("\n");
  printf("value: %d\n", value);

  // free allocated memory
  cudaFree(d_states);
  cudaFree(d_simScores);
  free(h_simScores);

  //return the average value
  return value /= sims;

}//end cornDogSetup function
