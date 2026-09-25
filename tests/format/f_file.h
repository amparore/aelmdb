/* f_file.h - raw access to an LMDB data file for the 03-aggformat tests.
 * Include after mdb.c; define DIR (the environment directory) first. */
static unsigned failures;

static void
fail(const char *what)
{
	fprintf(stderr, "FAIL %s\n", what);
	failures++;
}

static unsigned char *fbuf;
static size_t fsize;
static unsigned psize;
static void
load(void)
{
	FILE *f = fopen(DIR "/data.mdb", "rb");
	if (!f) { perror("open"); exit(2); }
	fseek(f, 0, SEEK_END);
	fsize = (size_t)ftell(f);
	fseek(f, 0, SEEK_SET);
	free(fbuf);
	fbuf = malloc(fsize);
	if (fread(fbuf, 1, fsize, f) != fsize) { perror("read"); exit(2); }
	fclose(f);
}

static void
store(void)
{
	FILE *f = fopen(DIR "/data.mdb", "r+b");
	if (!f || fwrite(fbuf, 1, fsize, f) != fsize) { perror("write"); exit(2); }
	fclose(f);
}

static MDB_page *
pg(pgno_t n)
{
	if ((size_t)(n + 1) * psize > fsize) { fprintf(stderr, "page %zu beyond file\n", (size_t)n); exit(2); }
	return (MDB_page *)(fbuf + (size_t)n * psize);
}

static MDB_meta *
meta(void)
{
	MDB_meta *m0 = (MDB_meta *)METADATA(pg(0)), *m1;
	psize = m0->mm_psize;
	m1 = (MDB_meta *)METADATA(pg(1));
	return m1->mm_txnid > m0->mm_txnid ? m1 : m0;
}

