CC = gcc
CFLAGS = -g -O0 -pedantic -Wall -std=c99 -Werror-implicit-function-declaration -Wunused-parameter
SOURCE = randomnumbers.c init.c mainloop.c
HEADER = randomnumbers.h types.h init.h
OBJECTS = $(SOURCE:.c=.o)
LDFLAGS = -lm


%.o : %.c $(HEADER)

.PHONY: clean

all: prog

prog: $(OBJECTS)
	$(CC) $(OBJECTS) $(CFLAGS) $(LDFLAGS) -o $@ 

clean:
	 rm -rf *.o *~
