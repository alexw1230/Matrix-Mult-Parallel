CC      = mpicc
CFLAGS  = -O3 -march=native -std=c11 -Wall -Wextra
LDLIBS  = -lm

SRC   = src
BUILD = build
OBJS  = $(BUILD)/main.o $(BUILD)/mm_ring.o $(BUILD)/kernel.o

mm: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $@

$(BUILD)/main.o:    $(SRC)/main.c $(SRC)/kernel.h $(SRC)/mm_ring.h
$(BUILD)/mm_ring.o: $(SRC)/mm_ring.c $(SRC)/mm_ring.h $(SRC)/kernel.h
$(BUILD)/kernel.o:  $(SRC)/kernel.c $(SRC)/kernel.h

clean:
	rm -rf mm $(BUILD)

.PHONY: clean