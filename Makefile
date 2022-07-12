CC=gcc
CFLAGS=-g -lm -fopenmp -std=c11

main: main.o 
	$(CC) $(CFLAGS) -o main main.o 

main_par_layer: main_par_layer.o
	$(CC) $(CFLAGS) -o main_par_layer main_par_layer.o

