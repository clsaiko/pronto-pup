/* file:	cornDogModule.c
 * desc:	A corn dog stand simulation function to include within Python.
 * date:	12/6/18
 * name:	Chris Saiko */

#define _XOPEN_SOURCE 500

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <semaphore.h>

#include <Python.h>       //for calling C module from Python
#include "pppSim.h"       //headers and structs

int cornDogSim(struct simArgs*);

//from python documentation
// static PyObject *
// spam_system(PyObject *self, PyObject *args)
// {
//     const char *command;
//     int sts;
//
//     if (!PyArg_ParseTuple(args, "s", &command))
//         return NULL;
//     sts = system(command);
//     return Py_BuildValue("i", sts);
// }

//from python documentation
static PyObject *
spam_cornDogSim(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

		// create the simulation arguments
		struct simArgs* pyargs = (struct simArgs*) malloc(sizeof(struct simArgs));

		// Is this the way to properly do this? Maybe I should take software engineering
		// args is a tuple with 41(!) arguments of the form:
		// { 	<numFryers>, <maxFryerSize>, <numCustQueues>, <openHour>, <closeHour>,
		// 		<corndogPrice>, <corndogCost>, <corndogCookTime>, <wage>, <customerPerMin>,
		//		<drinkChance>, <drinkTempChance>, <foodChance>, <foodMealChance>,
		//		<customerMaxQueueTime>, <customerChance>, <customerChanceHour>,
		//		<h0>, <h1>, <h2>, <h3>, <h4>, <h5>, <h6>, <h7>, <h8>,
		//		<h9>, <h10>, <h11>, <h12>, <h13>, <h14>, <h15>, <h16>,
		//		<h17>, <h18>, <h19>, <h20>, <h21>, <h22>, <h23> }

		// parse the python tuple to get the arguments
    if (!PyArg_ParseTuple(args, "if", &pyargs->numFryers, &pyargs->maxFryerSize, &pyargs->numCustQueues, &pyargs->openHour,
			&pyargs->closeHour, &pyargs->corndogPrice, &pyargs->corndogCost, &pyargs->corndogCookTime, &pyargs->wage, &pyargs->customerPerMin,
			&pyargs->drinkChance, &pyargs->drinkTempChance, &pyargs->foodChance, &pyargs->foodMealChance, &pyargs->customerMaxQueueTime,
			&pyargs->customerChance, &pyargs->customerChanceHour, &pyargs->hourlyTemps[0], &pyargs->hourlyTemps[1], &pyargs->hourlyTemps[2],
			&pyargs->hourlyTemps[3], &pyargs->hourlyTemps[4], &pyargs->hourlyTemps[5], &pyargs->hourlyTemps[6], &pyargs->hourlyTemps[7],
			&pyargs->hourlyTemps[8], &pyargs->hourlyTemps[9], &pyargs->hourlyTemps[10], &pyargs->hourlyTemps[11], &pyargs->hourlyTemps[12],
			&pyargs->hourlyTemps[13], &pyargs->hourlyTemps[14], &pyargs->hourlyTemps[15], &pyargs->hourlyTemps[16], &pyargs->hourlyTemps[17],
			&pyargs->hourlyTemps[18], &pyargs->hourlyTemps[19], &pyargs->hourlyTemps[20], &pyargs->hourlyTemps[21], &pyargs->hourlyTemps[22],
			&pyargs->hourlyTemps[23]))
        return NULL;


    sts = cornDogSim(pyargs);
    return Py_BuildValue("i", sts);
}

// the corn dog simulation function
int cornDogSim(struct simArgs* args){

  struct stand* stand = (struct stand*)malloc(sizeof(struct stand));
  struct behavior* behavior = (struct behavior*)malloc(sizeof(struct behavior));

	//extract arguments to variables

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


/*
	printf("Passed Argument       Value(s)\n\n");
	printf("numFryers             %d\n", stand->numFryers);
	printf("maxFryerSize          %d\n", stand->maxFryerSize);
	printf("numCustQueues         %d\n", stand->numCustQueues);
	printf("openHour              %d\n", stand->openHour);
	printf("closeHour             %d\n", stand->closeHour);
	printf("corndogPrice          %f\n", stand->corndogPrice);
	printf("corndogCost           %f\n", stand->corndogCost);
	printf("corndogCookTime       %d\n", stand->corndogCookTime);
	printf("wage                  %f\n", stand->wage);
	printf("customerPerMin        %d\n", stand->customerPerMin);

	printf("drinkChance           %f\n", behavior->drinkChance);
	printf("drinkTempChance       %f\n", behavior->drinkTempChance);
	printf("foodChance            %f\n", behavior->foodChance);
	printf("foodMealChance        %f\n", behavior->foodMealChance);
	printf("customerMaxQueueTime  %d\n", behavior->customerMaxQueueTime);
	printf("customerChance        %f\n", behavior->customerChance);
	printf("customerChanceHour    %f\n", behavior->customerChanceHour);

	printf("Hourly temps:        ");
	for (t = 0; t < 23; t++){
		printf(" %d", stand->hourlyTemps[t]);
	}
	printf("\n\n");
*/

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
	//printf("sim ticks             %d\n",ticks);

  // set up temp pointers for navigating the buffers
  // struct customer* tempLine;
  // struct cornDog* tempFood;
  // struct cornDog* tempFryer;
  //
	// // testing data
	// //add three dogs to food buffer
	// struct cornDog* testDog;
	// int d;

  struct timeval time1;
  gettimeofday(&time1, NULL);
  srand(time1.tv_usec * time1.tv_sec);

	// simulate the corn dog stand
	for (i = 0; i < ticks; i++){

		// update variables
		stand->currentHour = stand->openHour + (i / 60);
		//printf("currentHour    %d\n",currentHour);

		// food buffer handler
    //printf("food handling\n");
    foods_handler(cookedFood, stand);
		//printf("foods list");
		//list_foods(cookedFood);

    // for each fryer
		// fryer buffer handler
    //printf("fryer handling\n");
    if (stand->numFryers >= 1) {
      fryer_handler(fryer1, cookedFood, stand);
			//printf("fryer 1 list");
			//list_foods(fryer1);
    }
    if (stand->numFryers == 2) {
      fryer_handler(fryer2, cookedFood, stand);
			//printf("fryer 2 list");
			//list_foods(fryer2);
    }


		//printf("line handling\n");
		// customer line buffer handler
    if (stand->numCustQueues == 1)	{
      custLine_handler(custLine1, cookedFood, stand, behavior);
  	}
  	if (stand->numCustQueues == 2) {
      custLine_handler(custLine1, cookedFood, stand, behavior);
      custLine_handler(custLine2, cookedFood, stand, behavior);
  	}

		//printf("new customer handler\n");
		// new customers handler
    if (stand->numCustQueues == 1)	{
      newCust_handler(custLine1, stand, behavior);
      //printf("cust1 line");
      //list_line(custLine1);
  	}
  	if (stand->numCustQueues == 2) {
      if (custLine1->length < custLine2->length) newCust_handler(custLine1, stand, behavior);
      else newCust_handler(custLine2, stand, behavior);
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

	stand->lostCustomers += stand->waitingCustomers;

  //TODO free fryers
  // free(fryer1);
  // free(fryer2);

  //TODO free customer lines
  // free(custLine1);
  // free(custLine2);

  //TODO free food buffer
  // free(cookedFood);
  //
  // free(stand);
  // free(behavior);

	// total wages paid
	stand->totalWages = (stand->closeHour - stand->openHour) * stand->wage * (stand->numFryers + stand->numCustQueues);

	// total sales for the day
	stand->profit = stand->corndogSales * (stand->corndogPrice - stand->corndogCost) - stand->cornDogsWasted * stand->corndogCost;
	stand->profit = stand->profit + stand->drinkSales * (stand->drinkPrice - stand->drinkCost);
	stand->profit -= stand->totalWages;

/*
	printf("Wages paid:           %f\n",stand->totalWages);
	printf("Total profit:         %f\n",stand->profit);
  printf("Wasted corn dogs      %d\n",stand->cornDogsWasted);
  printf("Corn dog sales        %d\n",stand->corndogSales);
  printf("Drink sales           %d\n",stand->drinkSales);
  printf("Lost customers        %d\n",stand->lostCustomers);
*/

	// calculate daily score, higher score is better
	stand->dailyScore = 1 + stand->servedCustomers * 3;
	stand->dailyScore = stand->dailyScore - (stand->lostCustomers * 2) - (stand->cornDogsWasted * 1);
	stand->dailyScore += (4 * stand->profit);

	return stand->dailyScore;
}
