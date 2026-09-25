#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "lmdb.h"
#ifndef QN
#define QN 200000u
#endif
#define CHK(x) do { int _r=(x); if(_r){fprintf(stderr,"%s:%d %s -> %s\n",__FILE__,__LINE__,#x,mdb_strerror(_r)); exit(1);} } while(0)
static double now(void){struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t rng=0x123456789abcdef0ULL; static uint64_t xr(void){rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return rng;}
static void mk(unsigned char *k, uint64_t x){for(int i=0;i<8;i++) k[i]=(unsigned char)(x>>(56-8*i));}
static volatile uint64_t sink;
int main(int argc,char **argv){
  unsigned N=argc>1?(unsigned)strtoul(argv[1],0,10):200000u;
  unsigned Q=argc>2?(unsigned)strtoul(argv[2],0,10):QN;
  const char *dir=argc>3?argv[3]:"qdb";
  char cmd[512]; snprintf(cmd,sizeof cmd,"rm -rf %s; mkdir -p %s",dir,dir); if(system(cmd)){}
  MDB_env *env; MDB_txn *txn; MDB_dbi dbi; CHK(mdb_env_create(&env)); CHK(mdb_env_set_mapsize(env,4ull<<30)); CHK(mdb_env_set_maxdbs(env,4)); CHK(mdb_env_open(env,dir,MDB_NOSYNC,0644));
  CHK(mdb_txn_begin(env,NULL,0,&txn)); CHK(mdb_dbi_open(txn,"d",MDB_CREATE|MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM,&dbi)); CHK(mdb_set_hash_offset(txn,dbi,0));
  unsigned char kb[8], vb[48]; MDB_val k={8,kb}, v={sizeof vb,vb}; memset(vb,0x5a,sizeof vb);
  for(unsigned i=0;i<N;){ unsigned stop=i+10000; if(stop>N)stop=N; for(;i<stop;i++){mk(kb,i); memcpy(vb,&i,sizeof i); CHK(mdb_put(txn,dbi,&k,&v,0));} if(i<N){CHK(mdb_txn_commit(txn)); CHK(mdb_txn_begin(env,NULL,0,&txn));} }
  CHK(mdb_txn_commit(txn)); CHK(mdb_txn_begin(env,NULL,MDB_RDONLY,&txn));
  MDB_agg a; uint64_t rank,di; double t0,t1; uint64_t s=0;
  t0=now(); for(unsigned i=0;i<Q;i++){CHK(mdb_agg_totals(txn,dbi,&a)); s+=a.mv_agg_entries;} t1=now(); printf("totals=%.6f ",t1-t0);
  t0=now(); for(unsigned i=0;i<Q;i++){uint64_t x=xr()%N; mk(kb,x); CHK(mdb_agg_prefix(txn,dbi,&k,NULL,0,&a)); s+=a.mv_agg_entries;} t1=now(); printf("prefix=%.6f ",t1-t0);
  unsigned char lb[8], hb[8]; MDB_val lo={8,lb}, hi={8,hb};
  t0=now(); for(unsigned i=0;i<Q;i++){uint64_t x=xr()%N,y=xr()%N; if(x>y){uint64_t z=x;x=y;y=z;} mk(lb,x);mk(hb,y);CHK(mdb_agg_range(txn,dbi,&lo,NULL,&hi,NULL,0,&a));s+=a.mv_agg_entries;} t1=now(); printf("range=%.6f ",t1-t0);
  t0=now(); for(unsigned i=0;i<Q;i++){uint64_t x=xr()%N;mk(kb,x);MDB_val kk=k, empty={0,NULL};CHK(mdb_agg_rank(txn,dbi,&kk,&empty,MDB_AGG_WEIGHT_KEYS,MDB_AGG_RANK_EXACT,&rank,&di));s+=rank;} t1=now(); printf("rank=%.6f ",t1-t0);
  MDB_val ok,od;
  t0=now(); for(unsigned i=0;i<Q;i++){uint64_t x=xr()%N;CHK(mdb_agg_select(txn,dbi,MDB_AGG_WEIGHT_ENTRIES,x,&ok,&od,&di));s+=ok.mv_size+od.mv_size;} t1=now(); printf("select=%.6f ",t1-t0);
  MDB_cursor *qcur; CHK(mdb_cursor_open(txn,dbi,&qcur)); t0=now(); for(unsigned i=0;i<Q;i++){uint64_t x=xr()%N;CHK(mdb_agg_cursor_seek_rank(qcur,x,&ok,&od));s+=ok.mv_size+od.mv_size;} t1=now(); printf("cursor_seek=%.6f ",t1-t0); mdb_cursor_close(qcur);
  uint64_t wlo=N/4, whi=(uint64_t)N*3/4; mk(lb,wlo);mk(hb,whi); MDB_agg_window w; memset(&w,0,sizeof w);
  CHK(mdb_agg_window_aggregate(txn,dbi,&lo,NULL,&hi,NULL,0,&w,0,1,&a)); uint64_t wsz=w.mv_abs_hi-w.mv_abs_lo;
  t0=now(); for(unsigned i=0;i<Q;i++){uint64_t x=xr()%(wsz?wsz:1);uint64_t e=x+64; if(e>wsz)e=wsz; CHK(mdb_agg_window_aggregate(txn,dbi,&lo,NULL,&hi,NULL,0,&w,x,e,&a));s+=a.mv_agg_entries;} t1=now(); printf("window_agg=%.6f ",t1-t0);
  t0=now(); for(unsigned i=0;i<Q;i++){uint64_t x=xr()%N;mk(kb,x);MDB_val empty={0,NULL};CHK(mdb_agg_window_rank(txn,dbi,&lo,NULL,&hi,NULL,0,&w,&k,&empty,&rank));s+=rank;} t1=now(); printf("window_rank=%.6f",t1-t0);
  sink=s; printf(" checksum=%llu\n",(unsigned long long)sink); mdb_txn_abort(txn); mdb_env_close(env); return 0;
}
