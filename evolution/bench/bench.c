/* Micro-benchmark shared by LMDB variants (same public API).
 *
 *   cc -std=c11 -O2 -D_GNU_SOURCE -DMDB_HASH_SIZE=32 -I<impl> \
 *      -DAGGF=<dbi agg flags, e.g. 0, 0x80, 0x380> -DHASHOFF=<1 if HASHSUM> \
 *      bench.c <impl>/mdb.o <impl>/midl.o -lpthread -o bench
 *   ./bench N [D] [dir]      D=0 plain DB, D>0 DUPSORT with D duplicates/key
 *
 * Phases: N random puts (txn of 10k), N random MDB_SET gets, full scan,
 * delete of every other key.  MDB_NOSYNC; timings are indicative only. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include "lmdb.h"
#ifndef AGGF
#define AGGF 0
#endif
#ifndef BENCH_VALUE_SIZE
#define BENCH_VALUE_SIZE 48
#endif
#define CHK(x) do{int _r=(x); if(_r){fprintf(stderr,"%s:%d %s -> %s\n",__FILE__,__LINE__,#x,mdb_strerror(_r)); exit(1);} }while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t rng=88172645463325252ull; static uint64_t xr(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static void mk(unsigned char *k,uint64_t x){for(int i=0;i<8;i++)k[i]=(unsigned char)(x>>(56-8*i));}
int main(int argc,char**argv){
  unsigned N=argc>1?atoi(argv[1]):500000, D=argc>2?atoi(argv[2]):0; /* D>0: DUPSORT with D dups per key */
  const char *dir=argc>3?argv[3]:"db";
  char cmd[512]; snprintf(cmd,sizeof cmd,"rm -rf %s; mkdir -p %s",dir,dir); if(system(cmd)){}
  MDB_env*env; MDB_txn*txn; MDB_dbi dbi; MDB_cursor*cur;
  CHK(mdb_env_create(&env)); CHK(mdb_env_set_mapsize(env,4ull<<30)); CHK(mdb_env_set_maxdbs(env,4)); CHK(mdb_env_open(env,dir,MDB_NOSYNC,0644));
  CHK(mdb_txn_begin(env,NULL,0,&txn)); CHK(mdb_dbi_open(txn,"d",MDB_CREATE|(D?MDB_DUPSORT:0)|AGGF,&dbi));
#if HASHOFF
  CHK(mdb_set_hash_offset(txn,dbi,0));
#endif
  CHK(mdb_txn_commit(txn));
  unsigned char kb[8], vb[BENCH_VALUE_SIZE]; MDB_val k={8,kb}, v={sizeof vb,vb};
  uint64_t *ids=malloc(sizeof(uint64_t)*N); for(unsigned i=0;i<N;i++) ids[i]=xr();
  double t0=now();
  for(unsigned i=0;i<N;){ CHK(mdb_txn_begin(env,NULL,0,&txn));
    for(unsigned j=0;j<10000&&i<N;j++,i++){ mk(kb,ids[i]);
      if(!D){ memset(vb,(int)i,sizeof vb); memcpy(vb,&ids[i],8); CHK(mdb_put(txn,dbi,&k,&v,0)); }
      else for(unsigned d=0;d<D;d++){ memset(vb,(int)(i+d),sizeof vb); vb[0]=(unsigned char)d; memcpy(vb+8,&ids[i],8); CHK(mdb_put(txn,dbi,&k,&v,0)); } }
    CHK(mdb_txn_commit(txn)); }
  double t1=now();
  /* random point reads */
  CHK(mdb_txn_begin(env,NULL,MDB_RDONLY,&txn)); CHK(mdb_cursor_open(txn,dbi,&cur));
  for(unsigned i=0;i<N;i++){ MDB_val kk={8,kb},vv; mk(kb,ids[(i*2654435761u)%N]); CHK(mdb_cursor_get(cur,&kk,&vv,MDB_SET)); }
  double t2=now();
  /* full scan */
  { MDB_val kk,vv; unsigned long c=0; int rc; for(rc=mdb_cursor_get(cur,&kk,&vv,MDB_FIRST);rc==0;rc=mdb_cursor_get(cur,&kk,&vv,MDB_NEXT)) c++; if(c!=(unsigned long)N*(D?D:1)) {fprintf(stderr,"scan count %lu\n",c); return 1;} }
  double t3=now(); mdb_cursor_close(cur); mdb_txn_abort(txn);
  /* delete half the keys */
  for(unsigned i=0;i<N;){ CHK(mdb_txn_begin(env,NULL,0,&txn));
    for(unsigned j=0;j<5000&&i<N;j++,i+=2){ mk(kb,ids[i]); CHK(mdb_del(txn,dbi,&k,NULL)); }
    CHK(mdb_txn_commit(txn)); }
  double t4=now();
  printf("put=%.3f get=%.3f scan=%.3f del=%.3f\n",t1-t0,t2-t1,t3-t2,t4-t3);
  mdb_env_close(env); return 0; }
