/* file:	pppResults.c
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
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <semaphore.h>

//#include <Python.h>       //for calling C module from Python
#include "cornDogModule.c" //the cornDogSim function
//#include "pppSim.h"       //headers and structs

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
		printf("Program Usage: $ ./pppResults <search type> <sim iterations> <time limit in seconds>\n");
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

	printf("Average value from corn dog sim: %d\n",value);

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

	// create and initialize the simulation arguments
  struct simArgs* args = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(args);

	// best args found
	struct simArgs* bestArgs = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(bestArgs);

	//create server logfile
	char logfilePath[MAX_FILE_NAME_SIZE];
	char logLine[MAX_IO_BUFFER_SIZE];
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

	// initially populate the corn dog stand variables
  // args->numFryers = 1;						// 1-2
  // args->maxFryerSize = 3;					// 3-6
	// args->numCustQueues = 1;				// 1-2
  // args->openHour = 5;							// 5-7
  // args->closeHour = 21;						// 21-23
  // args->corndogPrice = 2.00;			// 2.00-4.00, 0.25 increment
  // args->corndogCost = 1.00;				// 1.00-2.00, 0.25 increment
  // args->corndogCookTime = 3;			// 3-4
  // args->wage = 7.75;							// 7.75-9.00, 0.25 increment
  // args->customerPerMin = 2; 			//not used?
	//
  // args->drinkChance = 0.4;				// 0.4-0.8, 0.1 increment
  // args->drinkTempChance = 0.1;		// 0.1-0.2, 0.05 increment
  // args->foodChance = 0.5;					// 0.5-0.9, 0.1 increment
  // args->foodMealChance = 0.1;			// 0.1-0.2, 0.05 increment
  // args->customerMaxQueueTime = 5;	// 5-7
  // args->customerChance = 0.5;			// 0.5-0.8, 0.05 increment
  // args->customerChanceHour = 0.05;// 0.05-0.15, 0.05 increment

	int value = 0;
	int s = 0;
	int bestValue = 0;
	int iters = 0;

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

	// set the hourly temperature for the day
	int t;
	int hourlyTemps[24];
	for (t = 0; t < 23; t++){
		hourlyTemps[t] = args->hourlyTemps[t];
	}

  int customerPerMin = args->customerPerMin;	//unused

  // customer behaviors
  float drinkChance = args->drinkChance;
  float drinkTempChance = args->drinkTempChance;
  float foodChance = args->foodChance;
  float foodMealChance = args->foodMealChance;
  int customerMaxQueueTime = args->customerMaxQueueTime;
  float customerChance = args->customerChance;
  float customerChanceHour = args->customerChanceHour;

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

																			value = 0;
																			iters++;
																			//simulation
																			//set arguments
																			// corndog stand variables
																			// args->numFryers = numFryers;
																			// args->maxFryerSize = maxFryerSize;
																			// args->numCustQueues = numCustQueues;
																			// args->openHour = openHour;
																			// args->closeHour = closeHour;
																			// args->corndogPrice = corndogPrice;
																			// args->corndogCost = corndogCost;
																			//args->corndogCookTime = corndogCookTime;
																			//args->wage = wage;
																			// customer behaviors
																			// args->drinkChance = drinkChance;
																			//args->drinkTempChance = drinkTempChance;
																			//args->foodChance = foodChance;
																			//args->foodMealChance = foodMealChance;
																			//args->customerMaxQueueTime = customerMaxQueueTime;
																			//args->customerChance = customerChance;
																			//args->customerChanceHour = customerChanceHour;
																			//printf("Customer chance hour: %f\n", customerChanceHour);

																			//print_args(args);
																			for ( s = 0; s < sims; s++){
																				// sim with these arguments
																				value += cornDogSim(args);
																			}//end sim for
																			value /= sims;
																			//printf("SIM VALUE %d\n",value);
																			if (value > bestValue){
																				//save new best value
																				bestValue = value;

																				//save best arguments
																				// corndog stand variables
																				// bestArgs->numFryers = args->numFryers;
																				// bestArgs->maxFryerSize = args->maxFryerSize;
																				// bestArgs->numCustQueues = args->numCustQueues;
																				// bestArgs->openHour = args->openHour;
																				// bestArgs->closeHour = args->closeHour;
																				// bestArgs->corndogPrice = args->corndogPrice;
																				// bestArgs->corndogCost = args->corndogCost;
																				// bestArgs->corndogCookTime = args->corndogCookTime;
																				// bestArgs->wage = args->wage;
																				// // customer behaviors
																				// bestArgs->drinkChance = args->drinkChance;
																				// bestArgs->drinkTempChance = args->drinkTempChance;
																				// bestArgs->foodChance = args->foodChance;
																				// bestArgs->foodMealChance = args->foodMealChance;
																				// bestArgs->customerMaxQueueTime = args->customerMaxQueueTime;
																				// bestArgs->customerChance = args->customerChance;
																				// bestArgs->customerChanceHour = args->customerChanceHour;

																				best_args(args, bestArgs);
																				printf("New best args: %d Iters: %d\n",value, iters);
																				//print_args(bestArgs);

																			}//end if
																			if ((time(0) - time_start) > seconds){ //time limit reached
																				//return value
																				printf("Time expired: %d seconds\n", seconds);
																				printf("Iterations: %d\n", iters);
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
																				printf("Time expired: %d seconds\n", seconds);
																				printf("Best value: %d\n", bestValue);
																				printf("Best args found: \n");
																				print_args(bestArgs);

																				dup2(oldStdout,1);
																				close(logFile);

																				return bestValue;
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

	//probably won't ever get here
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
	printf("Best value: %d\n", bestValue);
	printf("Best args found: \n");
	print_args(bestArgs);

	dup2(oldStdout,1);
	close(logFile);

	return bestValue;
}


int DFS_search(int sims){

	// create and initialize the simulation arguments
  struct simArgs* args = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(args);

	// run simulations
  int value = 0;
	int s = 0;
	for ( s = 0; s < sims; s++){
		value += cornDogSim(args);
	}

	value /= sims;
	return value;
}


int LS_hillClimb(int sims, int seconds){

	// start time
	int time_start = time(0);
	int current_time = time(0);

	// create and initialize the simulation arguments
  struct simArgs* args = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(args);

	// best args found
	struct simArgs* bestArgs = (struct simArgs*) malloc(sizeof(struct simArgs));
	init_args(bestArgs);

	//create server logfile
	char logfilePath[MAX_FILE_NAME_SIZE];
	char logLine[MAX_IO_BUFFER_SIZE];
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

	int value = 0;
	int s = 0;
	int bestValue = 0;
	int iters = 0;

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

	// set the hourly temperature for the day
	int t;
	int hourlyTemps[24];
	for (t = 0; t < 23; t++){
		hourlyTemps[t] = args->hourlyTemps[t];
	}

  int customerPerMin = args->customerPerMin;	//unused

  // customer behaviors
  float drinkChance = args->drinkChance;
  float drinkTempChance = args->drinkTempChance;
  float foodChance = args->foodChance;
  float foodMealChance = args->foodMealChance;
  int customerMaxQueueTime = args->customerMaxQueueTime;
  float customerChance = args->customerChance;
  float customerChanceHour = args->customerChanceHour;

	while((time(0) - time_start) < seconds){

		value = 0;	//reset value
		iters++;
		for ( s = 0; s < sims; s++){
			// sim with these arguments
			value += cornDogSim(args);
		}//end sim for
		value /= sims;
		//printf("SIM VALUE %d\n",value);
		if (value > bestValue){
			//save new best value
			bestValue = value;
			//save best arguments
			best_args(args, bestArgs);
			//printf("New best args: \n");
		}
		else{
			best_args(bestArgs, args);	//go back to other args
			find_neighbor(args);			//find new neighbor
		}

	}//end time while


	//return value
	printf("Time expired: %d seconds\n", seconds);
	printf("Iterations %d\n",iters);
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
	printf("Time expired: %d seconds\n", seconds);
	printf("Iterations %d\n",iters);
	printf("Best value: %d\n", bestValue);
	printf("Best args found: \n");
	print_args(bestArgs);

	dup2(oldStdout,1);
	close(logFile);

	return bestValue;
}


// find_neighbor function
// This function finds a neighbor state by changing one of the 16 values of the
// argument at random
void find_neighbor(struct simArgs* args){

  // args->numFryers = 1;						// 1-2
  // args->maxFryerSize = 3;					// 3-6
	// args->numCustQueues = 1;				// 1-2
  // args->openHour = 5;							// 5-7
  // args->closeHour = 21;						// 21-23
  // args->corndogPrice = 2.00;			// 2.00-4.00, 0.25 increment
  // args->corndogCost = 1.00;				// 1.00-2.00, 0.25 increment
  // args->corndogCookTime = 3;			// 3-4
  // args->wage = 7.75;							// 7.75-9.00, 0.25 increment
  // args->drinkChance = 0.4;				// 0.4-0.8, 0.1 increment
  // args->drinkTempChance = 0.1;		// 0.1-0.2, 0.05 increment
  // args->foodChance = 0.5;					// 0.5-0.9, 0.1 increment
  // args->foodMealChance = 0.1;			// 0.1-0.2, 0.05 increment
  // args->customerMaxQueueTime = 5;	// 5-7
  // args->customerChance = 0.5;			// 0.5-0.8, 0.05 increment
  // args->customerChanceHour = 0.05;// 0.05-0.15, 0.05 increment

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
	int s = 0;
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
	char logLine[MAX_IO_BUFFER_SIZE];
	char str[16];
	strcpy(logfilePath,"");
	strcpy(logfilePath,"SimAnnealing_");
	// sprintf(str, "%f", alpha);
	// strcat(logfilePath,str);
	// strcat(logfilePath,"sec");
	strcat(logfilePath,".log");
	int logFile = open(logfilePath, O_CREAT | O_TRUNC, 0666);
	if (logFile == -1){
		//failed to open file
		perror("Failed to open log file.");
		return -1;
	}//end file check if
	close(logFile);


	//do an initial sim on the initial arguments
	for ( s = 0; s < sims; s++){
		value += cornDogSim(args);
	}
	value /= sims;
	bestValue = value;	//equal at this point


	while(temperature > end){
		iters++;

		find_neighbor(args);			//find new neighbor

		// run simulation to get score value
		value = 0;
		for ( s = 0; s < sims; s++){
			value += cornDogSim(args);
		}
		value /= sims;

		delta = bestValue - value;
		if (delta < 0){
			//better score
			bestValue = value;
			best_args(args,bestArgs);
			printf("New best args: \n");
		}
		else{
			proba = (float)rand()/2147483648;	//kinda hacky, mersenne twister would be better
			// printf("proba: %f\n", proba);
			// printf("temp:  %f\n",temperature);
			// printf("delta: %d\n",delta);

			if (proba < exp(-delta/temperature) ){
				//keep worse value
				bestValue = value;
				best_args(args,bestArgs);
				printf("New worse best args: \n");
			}//end probability if
		}//end else

		// cooling process for temperature
		temperature *= alpha;

	}//end temperature while

	//write to log
	//return value
	//printf("Time expired: %d seconds\n", seconds);
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
