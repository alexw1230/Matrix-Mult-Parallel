CC      = mpicc
CFLAGS  = -O3 -march=native -std=c11 -Wall -Wextra
LDLIBS  = -lm

OBJS = main.o mm_ring.o kernel.o

mm: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

main.o:    main.c kernel.h mm_ring.h
mm_ring.o: mm_ring.c mm_ring.h kernel.h
kernel.o:  kernel.c kernel.h

clean:
	rm -f mm $(OBJS)

.PHONY: clean