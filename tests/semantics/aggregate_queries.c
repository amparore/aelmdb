#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "lmdb.h"

typedef struct Rec {
	MDB_val k, v;
	unsigned char *kb, *vb;
} Rec;

typedef struct Vec {
	Rec *v;
	size_t n, cap;
} Vec;

static void die_rc(const char *where, int rc) {
	if (rc) { fprintf(stderr, "%s: %d %s\n", where, rc, mdb_strerror(rc)); exit(2); }
}
static void check(int ok, const char *msg) {
	if (!ok) { fprintf(stderr, "FAIL: %s\n", msg); exit(3); }
}
static void vec_push(Vec *x, const MDB_val *k, const MDB_val *v) {
	Rec *r;
	if (x->n == x->cap) {
		x->cap = x->cap ? x->cap * 2 : 128;
		x->v = (Rec *)realloc(x->v, x->cap * sizeof(*x->v));
		check(x->v != NULL, "realloc vec");
	}
	r = &x->v[x->n++];
	r->kb = (unsigned char *)malloc(k->mv_size ? k->mv_size : 1);
	r->vb = (unsigned char *)malloc(v->mv_size ? v->mv_size : 1);
	check(r->kb && r->vb, "malloc record");
	memcpy(r->kb, k->mv_data, k->mv_size);
	memcpy(r->vb, v->mv_data, v->mv_size);
	r->k.mv_size = k->mv_size; r->k.mv_data = r->kb;
	r->v.mv_size = v->mv_size; r->v.mv_data = r->vb;
}
static void vec_free(Vec *x) {
	size_t i;
	for (i=0; i<x->n; ++i) { free(x->v[i].kb); free(x->v[i].vb); }
	free(x->v); memset(x, 0, sizeof(*x));
}
static void be32(unsigned char *p, uint32_t x) {
	p[0]=(unsigned char)(x>>24); p[1]=(unsigned char)(x>>16);
	p[2]=(unsigned char)(x>>8); p[3]=(unsigned char)x;
}
static int valcmp(const MDB_val *a, const MDB_val *b) {
	size_t n = a->mv_size < b->mv_size ? a->mv_size : b->mv_size;
	int c = memcmp(a->mv_data, b->mv_data, n);
	if (c) return c;
	return a->mv_size < b->mv_size ? -1 : a->mv_size > b->mv_size;
}
static size_t slice_start(size_t n, int off) {
	if (off >= 0) { check((size_t)off + MDB_HASH_SIZE <= n, "positive slice"); return (size_t)off; }
	{
		size_t back = (size_t)(-(off + 1));
		check(MDB_HASH_SIZE + back <= n, "negative slice");
		return n - MDB_HASH_SIZE - back;
	}
}
static void hadd(unsigned char *dst, const unsigned char *src) {
	unsigned carry=0; size_t i;
	for (i=0;i<MDB_HASH_SIZE;i++) { unsigned s=dst[i]+src[i]+carry; dst[i]=(unsigned char)s; carry=s>>8; }
}
static void expected(const Vec *x, int dupsort, int keyhash, int hashoff,
	const MDB_val *lk, const MDB_val *ld, int lincl,
	const MDB_val *hk, const MDB_val *hd, int hincl,
	MDB_agg *out)
{
	size_t i, last_key=(size_t)-1;
	memset(out,0,sizeof(*out));
	for(i=0;i<x->n;i++) {
		const Rec *r=&x->v[i]; int in=1, c;
		if (lk) {
			c=valcmp(&r->k,lk);
			if (dupsort && ld && c==0) c=valcmp(&r->v,ld);
			if (c<0 || (c==0 && !lincl)) in=0;
		}
		if (hk) {
			c=valcmp(&r->k,hk);
			if (dupsort && hd && c==0) c=valcmp(&r->v,hd);
			if (c>0 || (c==0 && !hincl)) in=0;
		}
		if (!in) continue;
		out->mv_agg_entries++;
		if (last_key==(size_t)-1 || valcmp(&x->v[last_key].k,&r->k)!=0) {
			out->mv_agg_keys++; last_key=i;
		}
		{
			const MDB_val *s=keyhash?&r->k:&r->v;
			hadd(out->mv_agg_hashes,(const unsigned char*)s->mv_data+slice_start(s->mv_size,hashoff));
		}
	}
}
static void expect_agg(const MDB_agg *got,const MDB_agg *exp,unsigned schema,const char *where) {
	if (got->mv_flags != schema || got->mv_agg_entries != exp->mv_agg_entries ||
		got->mv_agg_keys != exp->mv_agg_keys || memcmp(got->mv_agg_hashes,exp->mv_agg_hashes,MDB_HASH_SIZE)) {
		fprintf(stderr,"agg mismatch %s flags=%x/%x e=%llu/%llu k=%llu/%llu\n",where,
			got->mv_flags,schema,(unsigned long long)got->mv_agg_entries,(unsigned long long)exp->mv_agg_entries,
			(unsigned long long)got->mv_agg_keys,(unsigned long long)exp->mv_agg_keys); exit(4);
	}
}
static void snapshot(MDB_txn *txn,MDB_dbi dbi,Vec *x) {
	MDB_cursor *c; MDB_val k={0},v={0}; int rc;
	die_rc("cursor_open",mdb_cursor_open(txn,dbi,&c));
	for(rc=mdb_cursor_get(c,&k,&v,MDB_FIRST);rc==0;rc=mdb_cursor_get(c,&k,&v,MDB_NEXT)) vec_push(x,&k,&v);
	check(rc==MDB_NOTFOUND,"snapshot end"); mdb_cursor_close(c);
}
static void fill_value(unsigned char *p,size_t n,uint32_t a,uint32_t b) {
	size_t i; memset(p,0,n); be32(p,a); if(n>=8) be32(p+4,b);
	for(i=8;i<n;i++) p[i]=(unsigned char)(a*17u+b*29u+i*13u);
}
static void put_plain(MDB_txn *txn,MDB_dbi dbi,unsigned n,size_t vlen) {
	unsigned i; unsigned char kb[8], *vb=(unsigned char*)malloc(vlen); MDB_val k={8,kb},v={vlen,vb};
	check(vb!=NULL,"plain value alloc");
	for(i=0;i<n;i++) { memset(kb,0,8); be32(kb+4,i*2+1); fill_value(vb,vlen,i,7); die_rc("plain put",mdb_put(txn,dbi,&k,&v,0)); }
	free(vb);
}
static void put_dups(MDB_txn *txn,MDB_dbi dbi,unsigned keys,unsigned base_dups,size_t vlen) {
	unsigned i,j; unsigned char kb[8], *vb=(unsigned char*)malloc(vlen); MDB_val k={8,kb},v={vlen,vb};
	check(vb!=NULL,"dup value alloc");
	for(i=0;i<keys;i++) {
		unsigned nd=(i%9==0)?base_dups*8:1+(i%base_dups);
		memset(kb,0,8); be32(kb+4,i*3+2);
		for(j=0;j<nd;j++) { fill_value(vb,vlen,j+1,i+3); die_rc("dup put",mdb_put(txn,dbi,&k,&v,0)); }
	}
	free(vb);
}
static void verify_queries(MDB_txn *txn,MDB_dbi dbi,int dupsort,int keyhash,int hashoff,unsigned schema) {
	Vec x={0}; MDB_agg got,exp; size_t i;
	snapshot(txn,dbi,&x); check(x.n>20,"enough records");
	expected(&x,dupsort,keyhash,hashoff,NULL,NULL,0,NULL,NULL,0,&exp);
	die_rc("totals",mdb_agg_totals(txn,dbi,&got)); expect_agg(&got,&exp,schema,"totals");
	die_rc("range open",mdb_agg_range(txn,dbi,NULL,NULL,NULL,NULL,0,&got));
	expect_agg(&got,&exp,schema,"range open");

	for(i=0;i<x.n;i+=x.n/11+1) {
		unsigned f=(i&1)?MDB_AGG_PREFIX_INCL:0;
		expected(&x,dupsort,keyhash,hashoff,NULL,NULL,0,&x.v[i].k,dupsort?&x.v[i].v:NULL,!!f,&exp);
		die_rc("prefix record",mdb_agg_prefix(txn,dbi,&x.v[i].k,dupsort?&x.v[i].v:NULL,f,&got));
		expect_agg(&got,&exp,schema,"prefix record");
		if(dupsort) {
			expected(&x,dupsort,keyhash,hashoff,NULL,NULL,0,&x.v[i].k,NULL,!!f,&exp);
			die_rc("prefix key",mdb_agg_prefix(txn,dbi,&x.v[i].k,NULL,f,&got));
			expect_agg(&got,&exp,schema,"prefix key");
		}
	}

	for(i=0;i<12;i++) {
		size_t a=(i*7)%x.n, b=x.n-1-((i*11)%x.n), t; unsigned f=0;
		if(a>b){t=a;a=b;b=t;} if(i&1)f|=MDB_RANGE_LOWER_INCL; if(i&2)f|=MDB_RANGE_UPPER_INCL;
		expected(&x,dupsort,keyhash,hashoff,&x.v[a].k,dupsort?&x.v[a].v:NULL,!!(f&MDB_RANGE_LOWER_INCL),
			&x.v[b].k,dupsort?&x.v[b].v:NULL,!!(f&MDB_RANGE_UPPER_INCL),&exp);
		die_rc("range record",mdb_agg_range(txn,dbi,&x.v[a].k,dupsort?&x.v[a].v:NULL,
			&x.v[b].k,dupsort?&x.v[b].v:NULL,f,&got));
		expect_agg(&got,&exp,schema,"range record");
	}

	for(i=0;i<x.n;i+=x.n/17+1) {
		MDB_val k={0},v={0}; uint64_t di=999,rank=999;
		die_rc("select entries",mdb_agg_select(txn,dbi,MDB_AGG_WEIGHT_ENTRIES,i,&k,&v,&di));
		check(valcmp(&k,&x.v[i].k)==0 && valcmp(&v,&x.v[i].v)==0,"select entries value");
		k=x.v[i].k; v=dupsort?x.v[i].v:(MDB_val){0,NULL};
		die_rc("rank entries",mdb_agg_rank(txn,dbi,&k,&v,MDB_AGG_WEIGHT_ENTRIES,MDB_AGG_RANK_EXACT,&rank,&di));
		check(rank==i,"rank entries exact");
	}

	/* SET_RANGE rank: exact boundary and one synthetic key/data gap. */
	{
		MDB_val k=x.v[x.n/3].k, v=dupsort?x.v[x.n/3].v:(MDB_val){0,NULL};
		uint64_t rank=UINT64_MAX, di=UINT64_MAX;
		die_rc("rank set-range exact",mdb_agg_rank(txn,dbi,&k,&v,MDB_AGG_WEIGHT_ENTRIES,MDB_AGG_RANK_SET_RANGE,&rank,&di));
		check(rank==x.n/3,"rank set-range exact index");
		if (!dupsort && x.v[x.n/3].k.mv_size >= 8) {
			unsigned char gap[8]; MDB_val gk={8,gap}, gd={0,NULL};
			memcpy(gap,x.v[x.n/3].k.mv_data,8); gap[7]++;
			die_rc("rank set-range gap",mdb_agg_rank(txn,dbi,&gk,&gd,MDB_AGG_WEIGHT_ENTRIES,MDB_AGG_RANK_SET_RANGE,&rank,&di));
			check(rank==x.n/3+1,"rank set-range gap index");
		}
	}

	/* Distinct-key select/rank. */
	{
		size_t ki=0, pos=0;
		while(pos<x.n) {
			size_t first=pos; MDB_val k={0},v={0},empty={0,NULL}; uint64_t rank=999,di=999;
			if((ki%5)==0) {
				die_rc("select keys",mdb_agg_select(txn,dbi,MDB_AGG_WEIGHT_KEYS,ki,&k,&v,&di));
				check(valcmp(&k,&x.v[first].k)==0,"select key");
				k=x.v[first].k;
				die_rc("rank keys",mdb_agg_rank(txn,dbi,&k,&empty,MDB_AGG_WEIGHT_KEYS,MDB_AGG_RANK_EXACT,&rank,&di));
				check(rank==ki && di==0,"rank keys exact");
			}
			pos++; while(pos<x.n && valcmp(&x.v[pos].k,&x.v[first].k)==0)pos++; ki++;
		}
	}

	/* Existing-cursor rank seek and next. */
	{
		MDB_cursor *c; die_rc("seek cursor open",mdb_cursor_open(txn,dbi,&c));
		for(i=0;i<x.n;i+=x.n/13+1) {
			MDB_val k={0},v={0}; die_rc("cursor seek rank",mdb_agg_cursor_seek_rank(c,i,&k,&v));
			check(valcmp(&k,&x.v[i].k)==0 && valcmp(&v,&x.v[i].v)==0,"cursor seek value");
			if(i+1<x.n){ die_rc("cursor next",mdb_cursor_get(c,&k,&v,MDB_NEXT)); check(valcmp(&k,&x.v[i+1].k)==0 && valcmp(&v,&x.v[i+1].v)==0,"cursor next after seek"); }
		}
		mdb_cursor_close(c);
	}
	vec_free(&x);
}

int main(void) {
	char path[]="/tmp/aggq1XXXXXX"; MDB_env *env; MDB_txn *txn; MDB_dbi plain,dup,keydb; int rc; size_t vlen=MDB_HASH_SIZE+24;
	unsigned schema=MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	check(mkdtemp(path)!=NULL,"mkdtemp");
	die_rc("env_create",mdb_env_create(&env)); die_rc("mapsize",mdb_env_set_mapsize(env,64u*1024u*1024u));
	die_rc("maxdbs",mdb_env_set_maxdbs(env,8)); die_rc("env_open",mdb_env_open(env,path,0,0664));
	die_rc("txn begin",mdb_txn_begin(env,NULL,0,&txn));
	die_rc("open plain",mdb_dbi_open(txn,"plain",MDB_CREATE|schema,&plain));
	die_rc("hash plain",mdb_set_hash_offset(txn,plain,-1)); put_plain(txn,plain,900,vlen);
	die_rc("open dup",mdb_dbi_open(txn,"dup",MDB_CREATE|MDB_DUPSORT|schema,&dup));
	die_rc("hash dup",mdb_set_hash_offset(txn,dup,0)); put_dups(txn,dup,75,9,vlen);
	/* Key-source DB: keys must be large enough for every hash width. */
	die_rc("open keydb",mdb_dbi_open(txn,"keydb",MDB_CREATE|schema|MDB_AGG_HASHSOURCE_FROM_KEY,&keydb));
	die_rc("hash keydb",mdb_set_hash_offset(txn,keydb,0));
	{
		unsigned i; size_t klen=MDB_HASH_SIZE+8; unsigned char *kb=malloc(klen), vb[16]; MDB_val k={klen,kb},v={sizeof(vb),vb};
		check(kb!=NULL,"keydb alloc");
		for(i=0;i<120;i++){ fill_value(kb,klen,i,1); fill_value(vb,sizeof(vb),i,2); die_rc("keydb put",mdb_put(txn,keydb,&k,&v,0)); }
		free(kb);
	}
	die_rc("commit",mdb_txn_commit(txn));

	die_rc("read txn",mdb_txn_begin(env,NULL,MDB_RDONLY,&txn));
	die_rc("reopen plain",mdb_dbi_open(txn,"plain",0,&plain));
	die_rc("reopen dup",mdb_dbi_open(txn,"dup",0,&dup));
	die_rc("reopen keydb",mdb_dbi_open(txn,"keydb",0,&keydb));
	verify_queries(txn,plain,0,0,-1,schema);
	verify_queries(txn,dup,1,0,0,schema);
	verify_queries(txn,keydb,0,1,0,schema|MDB_AGG_HASHSOURCE_FROM_KEY);
	mdb_txn_abort(txn); mdb_env_close(env);
	{
		char data[512],lock[512]; snprintf(data,sizeof(data),"%s/data.mdb",path); snprintf(lock,sizeof(lock),"%s/lock.mdb",path); unlink(data); unlink(lock); rmdir(path);
	}
	(void)rc;
	printf("aggregate query tests passed (MDB_HASH_SIZE=%d)\n",MDB_HASH_SIZE);
	return 0;
}
