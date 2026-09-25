#!/usr/bin/env python3
"""Structural fingerprint of an LMDB data file (64-bit, 4 KiB pages).

Walks the live B+tree(s) of the most recent meta page and emits, level by
level, what is logically and structurally present: page type, number of
nodes, node flags, keys, inline data, overflow payloads (not page numbers)
and, recursively, named DBs and DUPSORT sub-trees/sub-pages.  Page numbers,
free space, free-list pages and stale bytes are ignored, so two engines that
build the same tree with a different page-allocation order produce the same
fingerprint.

Aggregate-format files (03 and later, format tag A335 in the meta version,
whose low 16 bits are MDB_HASH_SIZE) are recognised automatically: the larger
MDB_db record (md_keys, md_hash_offset, md_hashsum) is parsed, and branch
nodes of pages carrying aggregate bits skip their EVEN-padded prefix before
the key.  The aggregate schema is part of the walk; the aggregate values
(branch prefixes, md_keys, md_hashsum) are included too unless --no-agg-values
is given (in 03 they are opaque: zero for new links, carried by moves).

Usage:
  lmdb_fingerprint.py DATA.mdb [--even] [--dump] [--no-agg-values]
      --even   node data follows EVEN(ksize) (01-base and later); default on
      --odd    original LMDB 0.9.70 layout (data follows ksize)
      --dump   print the canonical walk instead of only its SHA-256
      --no-agg-values  aggregate format: leave out the aggregate values
"""
import hashlib
import struct
import sys

PS = 4096
P_BRANCH, P_LEAF, P_OVERFLOW, P_META, P_LEAF2, P_SUBP = 1, 2, 4, 8, 0x20, 0x40
F_BIGDATA, F_SUBDATA, F_DUPDATA = 1, 2, 4
HDR = 16          # page header: pgno(8) pad(2) flags(2) lower(2) upper(2)
NODESIZE = 8      # lo(2) hi(2) flags(2) ksize(2)
DBSZ = 48         # MDB_db: pad(4) flags(2) depth(2) branch/leaf/ovf(8x3) entries(8) root(8)
AGG_TAG = 0xA335  # aggregate format: meta version = AGG_TAG << 16 | MDB_HASH_SIZE
AGG_ENTRIES, AGG_KEYS, AGG_HASHSUM = 0x80, 0x100, 0x200
AGG_MASK = AGG_ENTRIES | AGG_KEYS | AGG_HASHSUM


def even(n):
    return (n + 1) & ~1


def align8(n):
    return (n + 7) & ~7


class Walker:
    def __init__(self, buf, even_keys=True, agg_values=True):
        self.b = buf
        self.even = even_keys
        self.out = []
        self.agg_values = agg_values
        version = struct.unpack_from('<I', self.page(0), HDR + 4)[0]
        # aggregate format: hash size from the version; MDB_db grows by
        # md_keys(8) md_hash_offset(2) md_hashsum(H), md_root 8-aligned
        self.hash = version & 0xffff if version >> 16 == AGG_TAG else 0
        if self.hash:
            self.root_off = align8(48 + 2 + self.hash)
            self.dbsz = self.root_off + 8
        else:
            self.root_off = 40
            self.dbsz = DBSZ

    def prefix(self, flags):
        """physical aggregate prefix of a branch node on a page with flags"""
        agg = flags & AGG_MASK
        raw = (8 if agg & AGG_ENTRIES else 0) + (8 if agg & AGG_KEYS else 0) + \
            (self.hash if agg & AGG_HASHSUM else 0)
        return raw, even(raw)

    def page(self, pgno):
        return self.b[pgno * PS:(pgno + 1) * PS]

    def emit(self, *x):
        self.out.append(repr(x))

    def nodes(self, p, base=0):
        lower = struct.unpack_from('<H', p, base + 12)[0]
        n = (lower - HDR) // 2
        for i in range(n):
            off = struct.unpack_from('<H', p, base + HDR + 2 * i)[0]
            yield base + off

    def db(self, rec, tag):
        pad, flags, depth, _b, _l, _o, entries = struct.unpack_from('<IHHQQQQ', rec, 0)
        root = struct.unpack_from('<Q', rec, self.root_off)[0]
        if self.hash:
            keys, hoff = struct.unpack_from('<Qh', rec, 40)
            self.emit('db', tag, flags, depth, entries, hoff)
            if self.agg_values:
                self.emit('dbagg', keys, rec[50:50 + self.hash])
        else:
            self.emit('db', tag, flags, depth, entries)
        if depth:
            self.tree(root, flags, depth)

    def tree(self, pgno, dbflags, depth, level=0):
        p = self.page(pgno)
        flags = struct.unpack_from('<H', p, 10)[0]
        if flags & P_LEAF2:
            lower = struct.unpack_from('<H', p, 12)[0]
            ksz = struct.unpack_from('<H', p, 8)[0]
            n = (lower - HDR) // 2
            self.emit('leaf2', level, n, p[HDR:HDR + n * ksz])
            return
        if flags & P_BRANCH:
            offs = list(self.nodes(p))
            raw, pfx = self.prefix(flags) if self.hash else (0, 0)
            if self.hash:
                self.emit('branch', level, len(offs), flags & AGG_MASK)
            else:
                self.emit('branch', level, len(offs))
            for i, o in enumerate(offs):
                lo, hi, nf, ks = struct.unpack_from('<HHHH', p, o)
                child = lo | (hi << 16) | (nf << 32)
                k0 = o + NODESIZE + pfx
                key = p[k0:k0 + ks] if i else b''
                self.emit('bkey', level, key)
                if raw and self.agg_values:
                    self.emit('bagg', p[o + NODESIZE:o + NODESIZE + raw])
                self.tree(child, dbflags, depth, level + 1)
            return
        if flags & P_LEAF:
            self.leaf(p, 0, level, dbflags)
            return
        self.emit('BADPAGE', level, flags)

    def leaf(self, p, base, level, dbflags):
        offs = list(self.nodes(p, base))
        self.emit('leaf', level, len(offs))
        for o in offs:
            lo, hi, nf, ks = struct.unpack_from('<HHHH', p, o)
            dsz = lo | (hi << 16)
            key = p[o + NODESIZE:o + NODESIZE + ks]
            doff = o + NODESIZE + (even(ks) if self.even else ks)
            if nf & F_BIGDATA:
                ov = struct.unpack_from('<Q', p, doff)[0]
                start = ov * PS + HDR
                self.emit('kv-ovf', key, nf, dsz, hashlib.sha256(self.b[start:start + dsz]).hexdigest())
            elif nf & F_SUBDATA:
                self.emit('kv-sub', key, nf)
                self.db(p[doff:doff + self.dbsz], key)
            elif nf & F_DUPDATA:
                self.emit('kv-subpage', key, nf, dsz)
                sub = p[doff:doff + dsz]
                sflags = struct.unpack_from('<H', sub, 10)[0]
                if sflags & P_LEAF2:
                    lower = struct.unpack_from('<H', sub, 12)[0]
                    ksz = struct.unpack_from('<H', sub, 8)[0]
                    n = (lower - HDR) // 2
                    self.emit('subleaf2', n, sub[HDR:HDR + n * ksz])
                else:
                    self.leaf(sub, 0, level + 1, 0)
            else:
                self.emit('kv', key, nf, p[doff:doff + dsz])

    def run(self):
        metas = []
        for m in (0, 1):
            p = self.page(m)
            txnid = struct.unpack_from('<Q', p, HDR + 4 + 4 + 8 + 8 + 2 * self.dbsz + 8)[0]
            metas.append((txnid, m))
        _, m = max(metas)
        p = self.page(m)
        dbs = HDR + 4 + 4 + 8 + 8
        self.db(p[dbs + self.dbsz:dbs + 2 * self.dbsz], b'MAIN')
        return self.out


def main():
    args = sys.argv[1:]
    even_keys = '--odd' not in args
    path = [a for a in args if not a.startswith('--')][0]
    out = Walker(open(path, 'rb').read(), even_keys, '--no-agg-values' not in args).run()
    if '--dump' in args:
        print('\n'.join(out))
    else:
        print(hashlib.sha256('\n'.join(out).encode()).hexdigest())


if __name__ == '__main__':
    main()
