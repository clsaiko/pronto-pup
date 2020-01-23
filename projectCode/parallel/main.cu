/* file:	main.cu
 * desc:	Returns the results of a search for a solution to the pronto pup problem.
 * date:	12/11/18
 * name:	Chris Saiko */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <semaphore.h>

#include "cornDogKernel.cu" //the cornDogSim function
//#include "support.h"       //headers and structs

#define NUM_ARGS 3

// search function definitions
int BFS_search(int sims, int seconds);
int DFS_search(int sims);
int LS_hillClimb(int sims, int seconds);
int LS_simAnn(int sims);
void find_neighbor(struct simArgs* args);

int main(int argc, char** argv) {

	printf("%s\n\n","EE 5351 Corn Dog Stand AI Search" );

	if (argc == 1) {
		//list program arguments
		printf("Program Usage: $ ./pppResults <search type> <sim iterations>  <time limit in seconds>\n");
		exit(1);
	}//end if

	if (argc != NUM_ARGS + 1){
    //Only expect three arguments
		printf("Wrong number of args, expected %d, given %d\n", NUM_ARGS, argc - 1);
		printf("Program Usage: $ ./pppResults <search type> <sim iterations> <time limit in seconds>\n");
		exit(1);
	}//end if

	int searchType = 0;
	if (strcmp(argv[1],"b") == 0) searchType = 1;
	if (strcmp(argv[1],"B") == 0) searchType = 1;
	if (strcmp(argv[1],"d") == 0) searchType = 2;
	if (strcmp(argv[1],"D") == 0) searchType = 2;
	if (strcmp(argv[1],"h") == 0) searchType = 7;
	if (strcmp(argv[1],"H") == 0) searchType = 7;
	if (strcmp(argv[1],"s") == 0) searchType = 8;
	if (strcmp(argv[1],"S") == 0) searchType = 8;
	if (searchType == 0){
		printf("Invalid search type %s\n",argv[1]);
		printf("b - Tree Search, h - Hill Climbing, s - Simulated Annealing.\n");
		exit(1);
	}

	int sims = atoi(argv[2]);
	if (sims < 1){
		printf("Number of simulations must be greater than 0.\n");
		exit(1);
	}

	int seconds = atoi(argv[3]);

	struct timeval time1;
	gettimeofday(&time1, NULL);
	srand(time1.tv_usec * time1.tv_sec);

/*
  printf("numFryers             %d\n", args->numFryers);
  printf("maxFryerSize          %d\n", args->maxFryerSize);
  printf("numCustQueues         %d\n", args->numCustQueues);
  printf("openHour              %d\n", args->openHour);
  printf("closeHour             %d\n", args->closeHour);
  printf("corndogPrice          %f\n", args->corndogPrice);
  printf("corndogCost           %f\n", args->corndogCost);
  printf("corndogCookTime       %d\n", args->corndogCookTime);
  printf("wage                  %f\n", args->wage);
  printf("customerPerMin        %d\n", args->customerPerMin);

  printf("drinkChance           %f\n", args->drinkChance);
  printf("drinkTempChance       %f\n", args->drinkTempChance);
  printf("foodChance            %f\n", args->foodChance);
  printf("foodMealChance        %f\n", args->foodMealChance);
  printf("customerMaxQueueTime  %d\n", args->customerMaxQueueTime);
  printf("customerChance        %f\n", args->customerChance);
  printf("customerChanceHour    %f\n", args->customerChanceHour);

  printf("Hourly temps:        ");
  for (t = 0; t < 23; t++){
    printf(" %d",args->hourlyTemps[t]);
  }
  printf("\n\n");
  */

	int value;
	printf("%d simulations per set of arguments.\n",sims);
	if (searchType != 8) printf("Running for maximum of %d seconds.\n",seconds);

	if (searchType == 1){
		printf("Method: Tree Search\n\n");
		value = BFS_search(sims, seconds);
		printf("\nTree Search\n");
	}

	if (searchType == 2){
		printf("Method: Tree Search\n\n");
		value = BFS_search(sims, seconds);
		printf("\nTree Search.\n");
	}

	if (searchType == 7){
		printf("Method: Hill Climbing\n\n");
		value = LS_hillClimb(sims, seconds);
		printf("\nHill Climbing Search.\n");
	}

	if (searchType == 8){
		printf("Method: Simulated Annealing\n\n");
		value = LS_simAnn(sims);
		printf("\nSimulated Annealing Search.\n");
	}

	//printf("Average value from corn dog sim: %d\n",value);

  //printf("customerMaxQueueTime  %d\n", args->customerMaxQueueTime);

  return 0;
}//end main

// BFS_search function
// searches over a solution tree to find the best value
// arguments are sims number of simulations for each state, and seconds a limit
// on how long to search the tree
int BFS_search(int sims, int seconds){

	// start time
	int time_start = time(0);
	int current_time = time(0);
	int value = 0;

	// create and initialize the simulation arguments
	struct simArgs* args = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(args);

	// best args found
	struct simArgs* bestArgs = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(bestArgs);

	//create server logfile
	char logfilePath[MAX_FILE_NAME_SIZE];
	char str[16];
	strcpy(logfilePath,"");
	strcpy(logfilePath,"search");
	sprintf(str, "%d", seconds);
	strcat(logfilePath,str);
	strcat(logfilePath,"sec");
	strcat(logfilePath,".log");
	int logFile = open(logfilePath, O_CREAT | O_TRUNC, 0666);
	if (logFile == -1){
		//failed to open file
		perror("Failed to open log file.");
		return -1;
	}//end file check if
	close(logFile);

	// allocate device memory for the arguments
	struct simArgs* d_args;
	cudaMalloc(&d_args, (sizeof(struct simArgs)));
	cudaMemset(d_args, 0, sizeof(struct simArgs));

	// allocate device memory for the simulation scores
	int * d_simScores;
	cudaMalloc(&d_simScores, sims * (sizeof(int)));
	cudaMemset(d_simScores, 0, sims * sizeof(int));

	// set heap size to 64 MB for in kernel malloc calls
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);

	// allocate host memory for the simulation scores
	int * h_simScores = (int *)malloc(sizeof(int) * sims);

	// copy data to device
	cudaMemcpy(d_args, args, sizeof(struct simArgs), cudaMemcpyHostToDevice);

	// CALL PARENT KERNEL
	tree_search_kernel<<<1,1>>>(sims, seconds, d_args, d_simScores);

	// copy data back
	cudaMemcpy(args, d_args, sizeof(struct simArgs), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_simScores, d_simScores, sizeof(int), cudaMemcpyDeviceToHost);

	value = h_simScores[0];
	/*
	printf("\nBest args found: \n");
	print_args(args);

	logFile = open(logfilePath, O_WRONLY | O_APPEND);
	if (logFile == -1){
		//failed to open file
		perror("Failed to open log file.");
		return -1;
	}//end file check if

	int oldStdout;
	oldStdout = dup(1);
	dup2(logFile,STDOUT_FILENO);
	printf("Best value: %d\n", value);
	printf("Best args found: \n");
	print_args(args);

	dup2(oldStdout,1);
	close(logFile);
	*/
	return value;
}


int DFS_search(int sims){

	// create and initialize the simulation arguments
  struct simArgs* args = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(args);

	// run simulations
  int value = 0;
	value = cornDogSim(sims, args);


	return value;
}


int LS_hillClimb(int sims, int seconds){

	// start time
	int time_start = time(0);
	int current_time = time(0);
	cudaError_t cudaRet;
	int value = 0;
	int bestValue = 0;

	// create and initialize the simulation arguments
	struct simArgs* args = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(args);

	// best args found
	struct simArgs* bestArgs = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(bestArgs);

	//create server logfile
	char logfilePath[MAX_FILE_NAME_SIZE];
	//char logLine[MAX_IO_BUFFER_SIZE];
	char str[16];
	strcpy(logfilePath,"");
	strcpy(logfilePath,"HillClimbing");
	sprintf(str, "%d", seconds);
	strcat(logfilePath,str);
	strcat(logfilePath,"sec");
	strcat(logfilePath,".log");
	int logFile = open(logfilePath, O_CREAT | O_TRUNC, 0666);
	if (logFile == -1){
		//failed to open file
		perror("Failed to open log file.");
		return -1;
	}//end file check if
	close(logFile);

	// allocate device memory for the arguments
	struct simArgs* d_args;
	cudaMalloc(&d_args, (sizeof(struct simArgs)));
	cudaMemset(d_args, 0, sizeof(struct simArgs));

	// allocate device memory for the simulation scores
	int * d_simScores;
	cudaMalloc(&d_simScores, sims * (sizeof(int)));
	cudaMemset(d_simScores, 0, sims * sizeof(int));

	// set heap size to 32 MB for in kernel malloc calls
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 32*1024*1024);

	// allocate host memory for the simulation scores
	int * h_simScores = (int *)malloc(sizeof(int) * sims);

	// copy data to device
	cudaMemcpy(d_args, args, sizeof(struct simArgs), cudaMemcpyHostToDevice);

	// CALL PARENT KERNEL
	hill_climbing_kernel<<<1,1>>>(sims, seconds, d_args, d_simScores);

	// copy data back
	cudaRet = cudaMemcpy(args, d_args, sizeof(struct simArgs), cudaMemcpyDeviceToHost);
	//if(cudaRet != 0) printf("Args copy error d->h");
	cudaRet = cudaMemcpy(h_simScores, d_simScores, sims * sizeof(int), cudaMemcpyDeviceToHost);
	//if(cudaRet != 0) printf("Score copy error d->h");

	value = h_simScores[0];
	/*
	printf("\nBest args found: \n");
	print_args(args);

	logFile = open(logfilePath, O_WRONLY | O_APPEND);
	if (logFile == -1){
		//failed to open file
		perror("Failed to open log file.");
		return -1;
	}//end file check if

	int oldStdout;
	oldStdout = dup(1);
	dup2(logFile,STDOUT_FILENO);
	printf("Best value: %d\n", value);
	printf("Best args found: \n");
	print_args(args);

	dup2(oldStdout,1);
	close(logFile);
	*/
	return value;

}//end LS_hillClimb



// find_neighbor host function
// This function finds a neighbor state by changing one of the 16 values of the
// argument at random
void find_neighbor(struct simArgs* args){

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
	float drinkChance = args->drinkChance;
	float drinkTempChance = args->drinkTempChance;
	float foodChance = args->foodChance;
	float foodMealChance = args->foodMealChance;
	int customerMaxQueueTime = args->customerMaxQueueTime;
	float customerChance = args->customerChance;
	float customerChanceHour = args->customerChanceHour;

	int randVar = rand() % 16;
	int varInt;
	float varFloat;

	if (randVar == 0){
		varInt = (rand() % 2) + 1;		//1-2
		while (varInt == numFryers){
			varInt = (rand() % 2) + 1;
		}
		args->numFryers = varInt;
		return;
	}

	if (randVar == 1){
		varInt = (rand() % 4) + 3;		//3-6
		while (varInt == maxFryerSize){
			varInt = (rand() % 3) + 3;
		}
		args->maxFryerSize = varInt;
		return;
	}

	if (randVar == 2){
		varInt = (rand() % 2) + 1;		//1-2
		while (varInt == numCustQueues){
			varInt = (rand() % 2) + 1;
		}
		args->numCustQueues = varInt;
		return;
	}

	if (randVar == 3){
		varInt = (rand() % 3) + 5;		//5-7
		while (varInt == openHour){
			varInt = (rand() % 3) + 5;
		}
		args->openHour = varInt;
		return;
	}

	if (randVar == 4){
		varInt = (rand() % 3) + 21;	//21-23
		while (varInt == closeHour){
			varInt = (rand() % 3) + 21;
		}
		args->closeHour = varInt;
		return;
	}

	if (randVar == 5){
		varFloat = (float)(rand() % 9)/4.0 + 2.00;	// 2.00-4.00, 0.25 increment
		while (varFloat == corndogPrice){
			varFloat = (float)(rand() % 9)/4.0 + 2.00;
		}
		args->corndogPrice = varFloat;
		return;
	}

	if (randVar == 6){
		varFloat = (float)(rand() % 5)/4.0 + 1.00;	// 1.00-2.00, 0.25 increment
		while (varFloat == corndogCost){
			varFloat = (float)(rand() % 5)/4.0 + 1.00;
		}
		args->corndogCost = varFloat;
		return;
	}

	if (randVar == 7){
		varInt = (rand() % 2) + 3;	// 3-4
		while (varInt == corndogCookTime){
			varInt = (rand() % 2) + 3;
		}
		args->corndogCookTime = varInt;
		return;
	}

	if (randVar == 8){
		varFloat = (float)(rand() % 6)/4.0 + 7.75;	// 7.75-9.00, 0.25 increment
		while (varFloat == wage){
			varFloat = (float)(rand() % 6)/4.0 + 7.75;
		}
		args->wage = varFloat;
		return;
	}

	if (randVar == 9){
		varFloat = (float)(rand() % 5)/10.0 + 0.4;	// 0.4-0.8, 0.1 increment
		while (varFloat == drinkChance){
			varFloat = (float)(rand() % 5)/10.0 + 0.4;
		}
		args->drinkChance = varFloat;
		return;
	}

	if (randVar == 10){
		varFloat = (float)(rand() % 2)/20.0 + 0.1;	// 0.1-0.2, 0.05 increment
		while (varFloat == drinkTempChance){
			varFloat = (float)(rand() % 2)/20.0 + 0.1;
		}
		args->drinkTempChance = varFloat;
		return;
	}

	if (randVar == 11){
		varFloat = (float)(rand() % 5)/10.0 + 0.5;	// 0.5-0.9, 0.1 increment
		while (varFloat == foodChance){
			varFloat = (float)(rand() % 5)/10.0 + 0.5;
		}
		args->foodChance = varFloat;
		return;
	}

	if (randVar == 12){
		varFloat = (float)(rand() % 2)/20.0 + 0.1;	// 0.1-0.2, 0.05 increment
		while (varFloat == foodMealChance){
			varFloat = (float)(rand() % 2)/20.0 + 0.1;
		}
		args->foodMealChance = varFloat;
		return;
	}

	if (randVar == 13){
		varInt = (rand() % 3) + 5;	// 5-7
		while (varInt == customerMaxQueueTime){
			varInt = (rand() % 3) + 5;
		}
		args->customerMaxQueueTime = varInt;
		return;
	}

	if (randVar == 14){
		varFloat = (float)(rand() % 7)/20.0 + 0.5;	// 0.5-0.8, 0.05 increment
		while (varFloat == customerChance){
			varFloat = (float)(rand() % 2)/20.0 + 0.1;
		}
		args->customerChance = varFloat;
		return;
	}

	if (randVar == 15){
		varFloat = (float)(rand() % 3)/20.0 + 0.05;	// 0.05-0.15, 0.05 increment
		while (varFloat == customerChanceHour){
			varFloat = (float)(rand() % 2)/20.0 + 0.1;
		}
		args->customerChanceHour = varFloat;
		return;
	}

}//end find_neighbor function


// LS_simAnn function
// This function utilizes simulated annealing local search to find a best set of
// variables within the corn dog stand simulation
// Based on a C# implementation from codeproject.com by Assaad Chalhoub
int LS_simAnn(int sims){

	// start time
	int time_start = time(0);
	int current_time = time(0);

	// create and initialize the simulation arguments
  struct simArgs* args = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(args);

	// best args found
	struct simArgs* bestArgs = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(bestArgs);

	int value = 0;
	int bestValue = 0;
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

	//create server logfile
	char logfilePath[MAX_FILE_NAME_SIZE];
	strcpy(logfilePath,"");
	strcpy(logfilePath,"SimAnnealing_");
	strcat(logfilePath,".log");
	int logFile = open(logfilePath, O_CREAT | O_TRUNC, 0666);
	if (logFile == -1){
		//failed to open file
		perror("Failed to open log file.");
		return -1;
	}//end file check if
	close(logFile);

	// allocate device memory for the arguments
	struct simArgs* d_args;
	cudaMalloc(&d_args, (sizeof(struct simArgs)));
	cudaMemset(d_args, 0, sizeof(struct simArgs));

	// allocate device memory for the simulation scores
	int * d_simScores;
	cudaMalloc(&d_simScores, sims * (sizeof(int)));
	cudaMemset(d_simScores, 0, sims * sizeof(int));

	// set heap size to 64 MB for in kernel malloc calls
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);

	// allocate host memory for the simulation scores
	int * h_simScores = (int *)malloc(sizeof(int) * sims);

	// copy data to device
	cudaMemcpy(d_args, args, sizeof(struct simArgs), cudaMemcpyHostToDevice);

	// CALL PARENT KERNEL
	sim_ann_kernel<<<1,1>>>(sims, d_args, d_simScores);

	// copy data back
	cudaMemcpy(args, d_args, sizeof(struct simArgs), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_simScores, d_simScores, sizeof(int), cudaMemcpyDeviceToHost);

	value = h_simScores[0];

	//write to log
	//return value
	printf("\nBest args found: \n");
	print_args(bestArgs);

	logFile = open(logfilePath, O_WRONLY | O_APPEND);
	if (logFile == -1){
		//failed to open file
		perror("Failed to open log file.");
		return -1;
	}//end file check if

	int oldStdout;
	oldStdout = dup(1);
	dup2(logFile,STDOUT_FILENO);
	printf("Time used:      %ld seconds\n", time(0) - time_start);
	printf("Alpha:          %f\n", output_alpha);
	printf("temperature:    %f\n", output_temperature);
	printf("end:            %f\n", output_end);
	printf("max iterations: %d\n", output_iterations);

	printf("iterations run: %d\n", iters);
	printf("Best value:     %d\n", bestValue);
	printf("Best args found: \n");
	print_args(bestArgs);

	dup2(oldStdout,1);
	close(logFile);

	return bestValue;

}
