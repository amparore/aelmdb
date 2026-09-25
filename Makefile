SHELL := /bin/sh
CC ?= cc
AR ?= ar
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -D_GNU_SOURCE
BUILD := build
SRC := src
HASHES ?= 32

.PHONY: all build test test-lmdb test-api test-semantics test-format test-regress \
        test-differential test-compare test-generic-hash test-sanitize verify-evolution bench clean check

all: build

build: $(BUILD)/liblmdb.a

$(BUILD)/liblmdb.a: $(SRC)/mdb.c $(SRC)/lmdb.h $(SRC)/midl.c $(SRC)/midl.h \
                    $(SRC)/mdb_agg_internal.h $(SRC)/mdb_agg_maint.c \
                    $(SRC)/mdb_agg_query.c $(SRC)/mdb_agg_debug.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -I$(SRC) -c $(SRC)/mdb.c -o $(BUILD)/mdb.o
	$(CC) $(CFLAGS) -I$(SRC) -c $(SRC)/midl.c -o $(BUILD)/midl.o
	$(AR) rcs $@ $(BUILD)/mdb.o $(BUILD)/midl.o

# Product-focused core battery. Heavy differential and upstream mtests remain
# explicit so normal development does not silently turn into a soak run.
test: test-semantics test-api test-format test-regress test-compare

test-lmdb:
	@HASHES="$(HASHES)" tests/run.sh lmdb

test-api:
	@HASHES="$(HASHES)" tests/run.sh api

test-semantics:
	@HASHES="$(HASHES)" tests/run.sh semantics

test-format:
	@HASHES="$(HASHES)" tests/run.sh format

test-regress:
	@HASHES="$(HASHES)" tests/run.sh regress

test-differential:
	@HASHES="$(HASHES)" tests/run.sh differential

test-compare:
	@HASHES="$(HASHES)" tests/run.sh compare

test-generic-hash:
	@tests/run.sh generic-hash

test-sanitize:
	@HASHES="$(HASHES)" tests/run.sh sanitize

verify-evolution:
	@evolution/tools/verify_evolution.sh

bench:
	@evolution/bench/run.sh quick

check: build verify-evolution test test-generic-hash

clean:
	rm -rf $(BUILD)
