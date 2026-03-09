# Makefile for AELMDB (Anti-Entropy Lightning memory-mapped database library)

########################################################################
# Toolchain / flags (override on command line if needed)

CC      ?= gcc
CXX     ?= g++
AR      ?= ar
RANLIB  ?= ranlib
RM      ?= rm -f
MKDIR_P ?= mkdir -p

THREADS ?= -pthread

# Keep the original warning set by default (can override WARN=...)
WARN    ?= -W -Wall -Wno-unused-parameter -Wbad-function-cast -Wuninitialized
# C++ doesn't like -Wbad-function-cast; drop it by default for C++ builds.
CXXWARN ?= $(filter-out -Wbad-function-cast,$(WARN))

# Default to debug-friendly options used in this repo.
# OPT      ?= -g -DMDB_DEBUG_AGG_PRINT -DMDB_DEBUG_AGG_INTEGRITY -DMDB_DEBUG
OPT      ?= -g -DMDB_DEBUG_AGG_INTEGRITY
# OPT     ?= -O3 -DNDEBUG
# OPT      ?= -g -pg

CPPFLAGS ?=
CFLAGS   ?= $(THREADS) $(OPT) $(WARN) $(XCFLAGS)
CXXFLAGS ?= $(THREADS) $(OPT) $(CXXWARN) $(XCFLAGS) -std=c++20
LDFLAGS  ?= #-g -pg
LDLIBS   ?=
SOLIBS   ?=

SOEXT   ?= .so

########################################################################
# Install locations

prefix      ?= /usr/local
exec_prefix ?= $(prefix)
bindir      ?= $(exec_prefix)/bin
libdir      ?= $(exec_prefix)/lib
includedir  ?= $(prefix)/include
datarootdir ?= $(prefix)/share
mandir      ?= $(datarootdir)/man

########################################################################
# Sources / build products

# Output folders (override on command line if needed: make OBJ=... BIN=...)
OBJ ?= obj
BIN ?= bin

IHDRS   := lmdb.h
IDOCS   := mdb_stat.1 mdb_copy.1 mdb_dump.1 mdb_load.1 mdb_drop.1

# Select the MDB implementation file (override: make MDB_SRC=mdb_ver30.c)
MDB_SRC ?= mdb.c

LMDB_OBJS := $(addprefix $(OBJ)/,mdb.o midl.o)
LMDB_LO   := $(addprefix $(OBJ)/,mdb.lo midl.lo)

LIB_STATIC := $(BIN)/liblmdb.a
LIB_SHARED := $(BIN)/liblmdb$(SOEXT)
ILIBS      := $(LIB_STATIC) $(LIB_SHARED)

# "Tools" (ship with LMDB)
# IPROGS := mdb_stat mdb_copy mdb_dump mdb_load mdb_drop
IPROGS :=

# Tests / local utilities
TEST_PROGS := mtest mtest2 mtest3 mtest4 mtest5 mtest_unit mtest_unit_keyhash mtest_adv mtest_unitxx mtest_perf

# Optional programs (only built if corresponding source exists)
OPTIONAL_PROGS := $(foreach p,mtest6 mplay,$(if $(wildcard $(p).c),$(p),))

PROGS := $(addprefix $(BIN)/,$(IPROGS) $(TEST_PROGS) $(OPTIONAL_PROGS))

TESTDB_DIRS := testdb testdb_agg_adv testdb_agg_keyhash testdb_agg_unit

.PHONY: all install clean test coverage

all: $(ILIBS) $(PROGS)

$(OBJ) $(BIN):
	$(MKDIR_P) $@

########################################################################
# Install / clean

install: $(ILIBS) $(addprefix $(BIN)/,$(IPROGS)) $(IHDRS)
	$(MKDIR_P) $(DESTDIR)$(bindir) $(DESTDIR)$(libdir) $(DESTDIR)$(includedir) $(DESTDIR)$(mandir)/man1
	for f in $(addprefix $(BIN)/,$(IPROGS)); do cp $$f $(DESTDIR)$(bindir); done
	for f in $(ILIBS);  do cp $$f $(DESTDIR)$(libdir); done
	for f in $(IHDRS);  do cp $$f $(DESTDIR)$(includedir); done
	for f in $(IDOCS);  do cp $$f $(DESTDIR)$(mandir)/man1; done

clean:
	$(RM) $(PROGS) $(ILIBS) *.o *.a *.so *~
	$(RM) -r $(OBJ) $(BIN)
	$(RM) -r $(TESTDB_DIRS) *.gcda *.gcno *.gcov xmtest x*

# Basic smoke test (matches the historical LMDB Makefile behavior)

test: all
	$(RM) -r $(TESTDB_DIRS)
	$(MKDIR_P) $(TESTDB_DIRS)
	./$(BIN)/mtest && ./$(BIN)/mdb_stat testdb

########################################################################
# Libraries

$(LIB_STATIC): $(LMDB_OBJS) | $(BIN)
	$(AR) rs $@ $^
	-$(RANLIB) $@ 2>/dev/null || true

$(LIB_SHARED): $(LMDB_LO) | $(BIN)
	$(CC) $(LDFLAGS) $(THREADS) -shared -o $@ $^ $(SOLIBS)

########################################################################
# Object files (MDB implementation is selected by MDB_SRC)

$(OBJ)/mdb.o: $(MDB_SRC) lmdb.h midl.h | $(OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $(MDB_SRC) -o $@

$(OBJ)/mdb.lo: $(MDB_SRC) lmdb.h midl.h | $(OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) -fPIC -c $(MDB_SRC) -o $@

$(OBJ)/midl.o: midl.c midl.h | $(OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(OBJ)/midl.lo: midl.c midl.h | $(OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) -fPIC -c $< -o $@

# Generic C compilation (for tools/tests)
$(OBJ)/%.o: %.c lmdb.h | $(OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

########################################################################
# Programs

# Tools and C tests link against the static library
$(BIN)/%: $(OBJ)/%.o $(LIB_STATIC) | $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# C++ unit test (requires lmdbxx-aelmdb)
$(OBJ)/mtest_unitxx.o: mtest_unit++.cpp ../lmdbxx-aelmdb/include/lmdbxx/lmdb++.h | $(OBJ)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I. -I../lmdbxx-aelmdb/include/ -c $< -o $@

$(BIN)/mtest_unitxx: $(OBJ)/mtest_unitxx.o $(LIB_STATIC) | $(BIN)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

########################################################################
# Coverage (gcov)

COV_FLAGS := -fprofile-arcs -ftest-coverage -O0
COV_OBJS  := xmdb.o xmidl.o

coverage: $(COV_OBJS)
	@set -e; \
	for i in mtest*.c [0-9]*.c; do \
		[ -f "$$i" ] || continue; \
		j=$${i%.c}; \
		$(CC) $(CPPFLAGS) $(CFLAGS) $(COV_FLAGS) -c "$$i" -o "x$$j.o"; \
		$(CC) $(THREADS) $(COV_FLAGS) -o "x$$j" "x$$j.o" $(COV_OBJS) $(LDLIBS); \
		$(RM) -r testdb; $(MKDIR_P) testdb; \
		./"x$$j"; \
	done
	gcov $(MDB_SRC)
	gcov midl.c

xmtest: $(OBJ)/mtest.o $(COV_OBJS)
	$(CC) $(THREADS) $(COV_FLAGS) -o $@ $^

xmdb.o: $(MDB_SRC) lmdb.h midl.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(COV_FLAGS) -c $(MDB_SRC) -o $@

xmidl.o: midl.c midl.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(COV_FLAGS) -c midl.c -o $@