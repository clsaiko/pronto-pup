/* file:	pppSim.h
 * desc:	Utility functions for a corn dog stand simulation.
 * date:	12/6/18
 * name:	Chris Saiko */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <string.h>
#include <time.h>

/* The maximum amount of bytes for a file name */
#define MAX_FILE_NAME_SIZE 255

/* The maximum amount of bytes for each I/O operation */
#define MAX_IO_BUFFER_SIZE 1024

/////////////////////////////////////////////////////////////////////////
/////       SOME STRUCTS TO PROCESS THE DATA
//////////////////////////////////////////////////////////////////////////

// represents the arguments passed to the simulation function, containing
// variables describing the corndog stand, and a group of behaviors
// demonstrated by customers
struct simArgs {

  // corndog stand variables
  int numFryers;
	int maxFryerSize;
	int numCustQueues;
  int openHour;
  int closeHour;
  float corndogPrice;
  float corndogCost;
  int corndogCookTime;
  float wage;
  int hourlyTemps[24];
  int customerPerMin;

  // customer behaviors
  float drinkChance;
  float drinkTempChance;
  float foodChance;
  float foodMealChance;
  int customerMaxQueueTime;
  float customerChance;
  float customerChanceHour;

} simArgs;


// represents variables within a corn dog stand
struct stand {

  int numFryers;
	int maxFryerSize;
	int numCustQueues;
  int openHour;
  int closeHour;
  float corndogPrice;
  float corndogCost;
  int corndogCookTime;
  float wage;
  int hourlyTemps[24];
  int customerPerMin;

  // set up other sim variables
  int cornDogsWasted;
  int lostCustomers;
  int waitingCustomers;
  int servedCustomers;
  int foodItems;
  int drinkSales;
  int corndogSales;
  float profit;
  float totalWages;
  int walkingPast;
  int currentHour;
  int dailyScore;

  float drinkPrice;
  float drinkCost;

} stand;


// represents customer behaviors
struct behavior {

  float drinkChance;
  float drinkTempChance;
  float foodChance;
  float foodMealChance;
  int customerMaxQueueTime;
  float customerChance;
  float customerChanceHour;

} behavior;


// represents a corn dog in the fryer or for consumption by a customer
struct cornDog {
  int age;
  int items;
  int max;
  struct cornDog* prev;
  struct cornDog* next;
} cornDog;


// represents a customer in line waiting to make a purchase
struct customer {
  int waitTime;
  int length;
  int dogsWanted;
  int drinksWanted;
  struct customer* prev;
  struct customer* next;
} customer;


// helper functions for the corn dog sim function

// fryer_handler function
// ages corn dogs, places in foods buffer
void foods_handler(struct cornDog* foods, struct stand* stand){

  //printf("FH: foods_handler\n");
  // for each corn dog in buffer
  struct cornDog* tempFood = foods;
  struct cornDog* nextFood;
  if (foods->next == NULL){
      //empty buffer, nothing to do
      //printf("FH: empty buffer\n");
  }
  else{
    tempFood = tempFood->next;  //go to next item
    while(tempFood != NULL){
      tempFood->age++;   // age corn dog
      nextFood = tempFood->next; //save next pointer
      // if corn dog age > 15, toss corn dog
      if (tempFood->age > 15) {
        stand->cornDogsWasted++;
        foods->items -= 1;
        if (tempFood->next != NULL) tempFood->next->prev = tempFood->prev;
        tempFood->prev->next = tempFood->next;
        free(tempFood);
      }//end if
      tempFood = nextFood;  //go to next item
    }//end while
  }//end else

  return;
}//end foods_handler function


// fryer_handler function
// ages corn dogs, places in foods buffer
void fryer_handler(struct cornDog* fryer, struct cornDog* foods, struct stand* stand){

  //printf("FH: fryer_handler\n");
  struct cornDog* tempFryer = fryer;
  struct cornDog* tempFood = foods;
  struct cornDog* nextFood;
  if (tempFryer->next == NULL){
    //empty fryer, nothing to do yet
  }//end if
  else{
    tempFryer = tempFryer->next;  //go to next item
    while(tempFryer != NULL){
      tempFryer->age++;      //age corn dog
      nextFood = tempFryer->next; //save next pointer
      // if corn dog done cooking
      if (tempFryer->age >= stand->corndogCookTime){
        //remove from fryer
        tempFryer->prev->next = tempFryer->next;
        if (tempFryer->next != NULL) tempFryer->next->prev = tempFryer->prev;
        fryer->items--;
        tempFood = foods;
        //add to food buffer
        while (tempFood->next != NULL) tempFood = tempFood->next; //go to end of food buffer
        tempFood->next = tempFryer; //insert into foodBuffer
        tempFryer->prev = tempFood;
        tempFryer->next = NULL;
        (foods->items)++;
      }//end if
      tempFryer = nextFood;  //next fryer item
    }//end while
  }//end else

  // add a corndog for cooking if there is fryer space
  while (fryer->items < fryer->max){
    struct cornDog* newDog = (struct cornDog*)malloc(sizeof(struct cornDog));
    newDog->age = 0;
    newDog->next = NULL;
    tempFryer = fryer;
    fryer->items++;
    while (tempFryer->next != NULL) tempFryer = tempFryer->next; //go to end of fryer
    tempFryer->next = newDog; //insert into fryer
    newDog->prev = tempFryer;
  }

  return;
}//end fryer_handler function


// custLine_handler function
// ages customers, services customers
void custLine_handler(struct customer* line, struct cornDog* foods, struct stand* stand, struct behavior* behav){

  struct customer* tempCustomer = line;
  struct cornDog* tempFood = foods;
  struct customer* nextCustomer = line;
  int d;

  // for each customer in line
  if (tempCustomer->next == NULL){
    //empty line, nothing to do
    //printf("empty customer line\n");
    return;
  }//end if
  else{
    //printf("\nAGE customers\n");
    tempCustomer = tempCustomer->next;
    //age each customer in line
    while (tempCustomer != NULL){
      tempCustomer->waitTime++;
      tempCustomer = tempCustomer->next;
    }//end while

    //printf("SERVICE customers\n");
    //printf("dogs available: %d\n",foods->items);
    tempCustomer = line->next;  //attempt to service first customer
    nextCustomer = tempCustomer->next;
    //printf("customer wants: %d\n",tempCustomer->dogsWanted);
    if (tempCustomer->dogsWanted < foods->items){ //dogs available
      //printf("sell corn dog\n");
      //remove dogs from food buffer
      stand->corndogSales += tempCustomer->dogsWanted;
      while(tempCustomer->dogsWanted > 0){
        tempFood = foods;
        while(tempFood->next != NULL) tempFood = tempFood->next;  //go to end of list
        tempFood->prev->next = NULL;
        foods->items--;
        free(tempFood);
        tempCustomer->dogsWanted--;
      }//end while
      stand->drinkSales += tempCustomer->drinksWanted; //add drink sales
      //remove customer from line
      //printf("remove serviced customer\n");
      line->next = nextCustomer;
      if (nextCustomer != NULL) nextCustomer->prev = line;
      free(tempCustomer);
      line->length--;
      stand->servedCustomers++; //increment served customers
    }

    //printf("\nREMOVE impatient customers\n");
    // remove impatient customers from line
    tempCustomer = line->next;
    while(tempCustomer != NULL){
      nextCustomer = tempCustomer->next;
      if (tempCustomer->waitTime > behav->customerMaxQueueTime){  //customer tired of waiting
        tempCustomer->prev->next = tempCustomer->next;
        if (tempCustomer->next != NULL) tempCustomer->next->prev = tempCustomer->prev;
        stand->lostCustomers++;
        //printf("LOST customer, length %d\n",line->length);
        line->length--;
        free(tempCustomer);
      }
      else{
        return;
      }
      tempCustomer = nextCustomer;  //next customer
    }//end while

  }//end else

  return;
}//end custLine_handler


// newCust_handler function
// adds new customers to the shortest line
void newCust_handler(struct customer* line, struct stand* stand, struct behavior* behav){

  //srand(time(0));

  struct customer* tempCustomer = line;
  struct customer* nextCustomer = line->next;

  // customers walking past
  int cStop = 0;
  int drink = 0;
  int dog = 0;
  struct customer* newCust;
  if ( (rand() % (int)(100 / behav->customerChance)) < 100) cStop++;
  if ( (rand() % (int)(100 / (behav->customerChanceHour * stand->currentHour) )) < 100) cStop++;

  int c;
  for (c = 0; c < cStop; c++){
    // see if a new customer will choose to order something
    // calculate drink chance
    if ( (rand() % (int)(100 / behav->drinkChance)) < 100) drink++;
    if ( (rand() % (int)(100 / (behav->drinkTempChance * (stand->hourlyTemps[stand->currentHour] - 20) ) ) ) < 100) drink++;
    // calculate food chance
    if ((stand->currentHour % 4) == 0){
      if ( (rand() % (int)(100 / behav->foodMealChance)) < 100) dog++;
    }
    if ( (rand() % (int)(100 / behav->foodChance)) < 100) dog++;

    if ((dog > 0) || (drink > 0)) { //customer wants to order
      newCust = (struct customer*)malloc(sizeof(struct customer));
      newCust->waitTime = 0;
      newCust->dogsWanted = dog;
      newCust->drinksWanted = drink;
      newCust->next = NULL;
      while(tempCustomer->next != NULL) tempCustomer = tempCustomer->next;  //back of the line
      tempCustomer->next = newCust;
      newCust->prev = tempCustomer;
      stand->waitingCustomers++;
      line->length++;
      //printf("\nNEW CUSTOMER\n");
    }//end if
    //reset variables
    dog = 0;
    drink = 0;
  }//end for

  return;
}//end newCust_handler


// list_foods function
// Prints out a food buffer passed as an argument
void list_foods(struct cornDog* foods){

  struct cornDog* tempFood = foods->next;

  printf("\nFood buffer\n");
  printf("Max Size: %d\n",foods->max);
  printf("Items:    %d\n",foods->items);

  while(tempFood != NULL){
    printf("dog age: %d\n", tempFood->age);
    tempFood = tempFood->next;
  }
  printf("\n");
  return;
}//end list_foods


// list_line function
// Prints out a customer line passed as an argument
void list_line(struct customer* line){

  struct customer* tempLine = line->next;

  printf("\nCustomer Line Length,Age: %d,%d\n",line->length,line->waitTime);

  while(tempLine != NULL){
    printf("customer wait:%d ", tempLine->waitTime);
    printf("dogs:%d ", tempLine->dogsWanted);
    printf("drinks:%d ", tempLine->drinksWanted);
    printf("prev:%d ", tempLine->prev);
    printf("next:%d\n", tempLine->next);

    if (tempLine->next == tempLine->prev){
      printf("ERROR \n\n");
      return;
    }
    tempLine = tempLine->next;
  }
  printf("\n");
  return;
}//end list_line

// remove_dog function
// removes a specific corn dog from a buffer
void remove_dog(struct cornDog* foods, struct cornDog* dog){

  struct cornDog* tempFood = dog;
  if (dog->next != NULL) dog->next->prev = dog->prev;
  dog->prev->next = dog->next;
  foods->items -= 1;
  free(dog);

  return;
}//end remove_dog function


// remove_a_dog function
// removes last corn dog from a buffer
void remove_a_dog(struct cornDog* foods){

  struct cornDog* tempFood = foods;
  while (tempFood->next != NULL) tempFood = tempFood->next; //go to end of food buffer
  tempFood->prev->next = NULL;
  foods->items -= 1;
  free(tempFood);

  return;
}//end remove_a_dog function


// add_dog function
// adds a corn dog to a buffer
void add_dog(struct cornDog* foods, struct cornDog* dog){

  struct cornDog* tempFood = foods;

  while (tempFood->next != NULL) tempFood = tempFood->next; //go to end of food buffer
  tempFood->next = dog;
  dog->prev = tempFood;
  dog->next = NULL;
  foods->items += 1;

  return;
}//end add_dog function


// helper functions for the pppResults application

// init_args function
// initializes initial sim arguments
void init_args(struct simArgs* args){

  // initially populate the corn dog stand variables
  args->numFryers = 1;						// 1-2
  args->maxFryerSize = 3;					// 3-6
	args->numCustQueues = 1;				// 1-2
  args->openHour = 5;							// 5-7
  args->closeHour = 21;						// 21-23
  args->corndogPrice = 2.00;			// 2.00-4.00, 0.25 increment
  args->corndogCost = 1.00;				// 1.00-2.00, 0.25 increment
  args->corndogCookTime = 3;			// 3-4
  args->wage = 7.75;							// 7.75-9.00, 0.25 increment
  args->customerPerMin = 2; 			//not used?

  args->drinkChance = 0.4;				// 0.4-0.8, 0.1 increment
  args->drinkTempChance = 0.1;		// 0.1-0.2, 0.05 increment
  args->foodChance = 0.5;					// 0.5-0.9, 0.1 increment
  args->foodMealChance = 0.1;			// 0.1-0.2, 0.05 increment
  args->customerMaxQueueTime = 5;	// 5-7
  args->customerChance = 0.5;			// 0.5-0.8, 0.05 increment
  args->customerChanceHour = 0.05;// 0.05-0.15, 0.05 increment

  // set the hourly temperature for the day
  // arbitrarily set, future work here
  int t;
  for (t = 0; t < 12; t++){
    args->hourlyTemps[t] = (60 + t);
  }
  for (; t < 23; t++){
    args->hourlyTemps[t] = (84 - t);
  }

  return;
}//end init_args function


// best_args function
// stores best value sim arguments from args into bestArgs
void best_args(struct simArgs* args, struct simArgs* bestArgs){

  //printf("new best args\n");
  // save the arguments as a "best" state
  bestArgs->numFryers = args->numFryers;
  bestArgs->maxFryerSize = args->maxFryerSize;
	bestArgs->numCustQueues = args->numCustQueues;
  bestArgs->openHour = args->openHour;
  bestArgs->closeHour = args->closeHour;
  bestArgs->corndogPrice = args->corndogPrice;
  bestArgs->corndogCost = args->corndogCost;
  bestArgs->corndogCookTime = args->corndogCookTime;
  bestArgs->wage = args->wage;
  bestArgs->customerPerMin = args->customerPerMin;

  bestArgs->drinkChance = args->drinkChance;
  bestArgs->drinkTempChance = args->drinkTempChance;
  bestArgs->foodChance = args->foodChance;
  bestArgs->foodMealChance = args->foodMealChance;
  bestArgs->customerMaxQueueTime = args->customerMaxQueueTime;
  bestArgs->customerChance = args->customerChance;
  bestArgs->customerChanceHour = args->customerChanceHour;

  //print_args(bestArgs);

  return;
}//end save_args function


// print_args function
// prints out sim arguments
void print_args(struct simArgs* args){

  int t;

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

  return;
}//end print_args function




// represents the hosts layout from an input mapping file in a linked list/tree
// struct hosts {
//
//   char hostname[MAX_IO_BUFFER_SIZE];
//   char address[16];
//   pthread_mutex_t* mutex;
//   struct hosts* next;
//
// } hosts;


// arguments passed to child threads when invoking child_thread(arg)
// struct args {
//
//   struct hosts* hosts;
//   char ID[MAX_IO_BUFFER_SIZE];
//   int serverType;
//   int sock_fd;
//   pthread_mutex_t* hosts_mutex;
//   pthread_mutex_t* mappinglog_mutex;
//   pthread_mutex_t* defaultLog_mutex;
//   struct sockaddr_in clientAddress;
// };



/**********************************
*
* Taken from Unix Systems Programming, Robbins & Robbins, p37
*
*********************************/
int makeargv(const char *s, const char *delimiters, char ***argvp) {
   int error;
   int i;
   int numtokens;
   const char *snew;
   char *t;

   if ((s == NULL) || (delimiters == NULL) || (argvp == NULL)) {
      errno = EINVAL;
      return -1;
   }
   *argvp = NULL;
   snew = s + strspn(s, delimiters);
   if ((t = malloc(strlen(snew) + 1)) == NULL)
      return -1;
   strcpy(t,snew);
   numtokens = 0;
   if (strtok(t, delimiters) != NULL)
      for (numtokens = 1; strtok(NULL, delimiters) != NULL; numtokens++) ;

   if ((*argvp = malloc((numtokens + 1)*sizeof(char *))) == NULL) {
      error = errno;
      free(t);
      errno = error;
      return -1;
   }

   if (numtokens == 0)
      free(t);
   else {
      strcpy(t,snew);
      **argvp = strtok(t,delimiters);
      for (i=1; i<numtokens; i++)
         *((*argvp) +i) = strtok(NULL,delimiters);
   }
   *((*argvp) + numtokens) = NULL;
   return numtokens;
}

/**********************************
*
* Taken from Unix Systems Programming, Robbins & Robbins, p38
*
*********************************/
void freemakeargv(char **argv) {
   if (argv == NULL)
      return;
   if (*argv != NULL)
      free(*argv);
   free(argv);
}

char *trimwhitespace(char *str) {
  char *end;
  // Trim leading space
  while(isspace((unsigned char)*str)) str++;

  if(*str == 0)  // All spaces?
    return str;

  // Trim trailing space
  end = str + strlen(str) - 1;

  while(end > str && isspace((unsigned char)*end)) end--;

  // Write new null terminator
  *(end+1) = 0;

  return str;
}
