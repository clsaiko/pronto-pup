/* file:	pppSim.c
 * desc:	A corn dog stand simulation.
 * date:	12/6/18
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
#include <semaphore.h>

//#include <Python.h>       //for calling C module from Python
#include "cornDogModule.c" //the cornDogSim function
//#include "pppSim.h"       //headers and structs

#define NUM_ARGS 41

int main(int argc, char** argv) {

	printf("%s\n\n","CSCI 4511 Corn Dog Stand Simulator" );

	if (argc == 1) {
		//list program arguments
		printf("Program Usage: $ ./pppSim <numFryers> <maxFryerSize> <numCustQueues> <openHour> <closeHour>\n");
		printf("  <corndogPrice> <corndogCost> <corndogCookTime> <wage> <customerPerMin> <drinkChance>\n");
		printf("  <drinkTempChance> <foodChance> <foodMealChance> <customerMaxQueueTime> <customerChance>\n");
		printf("  <customerChanceHour> <h0> <h1> <h2> <h3> <h4> <h5> <h6> <h7> <h8> <h9> <h10> <h11> <h12>\n");
		printf("  <h13> <h14> <h15> <h16> <h17> <h18> <h19> <h20> <h21> <h22> <h23>\n\n");
		printf("  <h0> to <h23> are hourly temperatures in degrees Fahrenheit\n");
		exit(1);
	}//end if

	if (argc != NUM_ARGS + 1) {
    //Only expect 41 arguments
		printf("Wrong number of args, expected %d, given %d\n", NUM_ARGS, argc - 1);
		printf("Program Usage: $ ./pppSim <numFryers> <maxFryerSize> <numCustQueues> <openHour> <closeHour>\n");
		printf("  <corndogPrice> <corndogCost> <corndogCookTime> <wage> <customerPerMin> <drinkChance>\n");
		printf("  <drinkTempChance> <foodChance> <foodMealChance> <customerMaxQueueTime> <customerChance>\n");
		printf("  <customerChanceHour> <h0> <h1> <h2> <h3> <h4> <h5> <h6> <h7> <h8> <h9> <h10> <h11> <h12>\n");
		printf("  <h13> <h14> <h15> <h16> <h17> <h18> <h19> <h20> <h21> <h22> <h23>\n\n");
		printf("  <h0> to <h23> are hourly temperatures in degrees Fahrenheit\n");
		exit(1);
	}//end if

  // create the simulation arguments
  struct simArgs* args = (struct simArgs*) malloc(sizeof(struct simArgs));

  // populate the corn dog stand variables
  args->numFryers = atoi(argv[1]);
  args->maxFryerSize = atoi(argv[2]);
	args->numCustQueues = atoi(argv[3]);
  args->openHour = atoi(argv[4]);
  args->closeHour = atoi(argv[5]);
  args->corndogPrice = atof(argv[6]);
  args->corndogCost = atof(argv[7]);
  args->corndogCookTime = atoi(argv[8]);
  args->wage = atof(argv[9]);
  args->customerPerMin = atoi(argv[10]);  //not used?

  args->drinkChance = atof(argv[11]);
  args->drinkTempChance = atof(argv[12]);
  args->foodChance = atof(argv[13]);
  args->foodMealChance = atof(argv[14]);
  args->customerMaxQueueTime = 5;
  args->customerChance = atof(argv[16]);
  args->customerChanceHour = atof(argv[17]);

  // set the hourly temperature for the day
  int t;
  for (t = 0; t < 23; t++){
    args->hourlyTemps[t] = atoi(argv[t + 18]);
  }

/*
  printf("Passed Argument       Value(s)\n");
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

	// run 30 simulations
  int value = 0;
	int s = 0;
	for ( s = 0; s < 30; s++){
		value += cornDogSim(args);
	}

	value /= 30;

  printf("Average value from corn dog sim: %d\n",value);

  //printf("customerMaxQueueTime  %d\n", args->customerMaxQueueTime);

  return 0;
}//end main
