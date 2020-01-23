CC=gcc
CFLAGS=-std=gnu99 -pthread
LIBS=-lm
DBFLAGS=-g

all: pppSim pppResults

pppSim: pppSim.c cornDogModule.c pppSim.h
	$(CC) $(CFLAGS) -I/usr/include/python2.7 -o pppSim pppSim.c -lpython2.7

pppResults: pppResults.c cornDogModule.c pppSim.h
	$(CC) $(CFLAGS) -I/usr/include/python2.7 -o pppResults pppResults.c -lm -lpython2.7

debugs: pppSim.c cornDogModule.c pppSim.h
	$(CC) $(CFLAGS) $(DBFLAGS) -I/usr/include/python2.7 -o pppSim pppSim.c -lpython2.7

clean:
	rm pppSim pppResults

runsim:
	./pppSim 1 3 1 5 21 2 1 3 8.75 2 0.4 0.15 0.6 0.15 7 0.75 0.15 60 60 60 61 62 63 64 65 66 67 68 69 71 73 75 76 77 76 75 74 73 72 71 70

runresults:
	./pppResults b 30 30
