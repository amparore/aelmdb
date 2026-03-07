// mtest_unit++.cpp - simple C++17 unit tests for lmdb++ (AELMDB fork)
//
// Focus: exercising the lmdb++ wrapper API, especially the aggregate/hash features:
//   - MDB_AGG_ENTRIES / MDB_AGG_KEYS / MDB_AGG_HASHSUM
//   - dbi::totals(), prefix(), range(), rank(), select()
//   - DUPSORT record-order semantics (key,data)
//   - mdb_set_hash_offset() semantics
//
// Build (C++17 required for std::string_view; adapt to your tree):
//   c++ -O2 -std=c++17 -I. mtest_unit++.cpp mdb.c midl.c -o mtest_unitpp
//   c++ -g  -std=c++17 -I. mtest_unit++.cpp mdb.c midl.c -o mtest_unitpp
//

#include "lmdbxx/lmdb++.h"

#ifndef MDB_AELMDB_VERSION
#error "This project requires the AELMDB fork of LMDB: lmdb.h must define MDB_AELMDB_VERSION."
#endif

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h> // getpid()

namespace test {

    struct failure : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    inline std::string where(const char* file, int line)
    {
        std::ostringstream oss;
        oss << file << ":" << line;
        return oss.str();
    }

#define REQUIRE(cond) do { \
  if (!(cond)) { \
    std::ostringstream _oss; \
    _oss << "REQUIRE(" #cond ") failed at " << test::where(__FILE__, __LINE__); \
    throw test::failure(_oss.str()); \
  } \
} while (0)

#define REQUIRE_EQ(a,b) do { \
  const auto _a = (a); \
  const auto _b = (b); \
  if (!((_a) == (_b))) { \
    std::ostringstream _oss; \
    _oss << "REQUIRE_EQ(" #a ", " #b ") failed at " << test::where(__FILE__, __LINE__) \
         << " : left=" << _a << " right=" << _b; \
    throw test::failure(_oss.str()); \
  } \
} while (0)

    inline std::string bytes_to_hex(const uint8_t* p, std::size_t n)
    {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (std::size_t i = 0; i < n; ++i) {
            oss << std::setw(2) << static_cast<unsigned>(p[i]);
        }
        return oss.str();
    }

    inline std::string to_string_copy(const std::string_view v)
    {
        return std::string(v.data(), v.size());
    }
    inline uint8_t byte_at(const std::string_view v, const std::size_t idx)
    {
        REQUIRE(v.size() > idx);
        return reinterpret_cast<const uint8_t*>(v.data())[idx];
    }
    // Deterministic pseudo-random 32-byte "hash" derived from a tag.
    inline std::array<uint8_t, MDB_HASH_SIZE> make_hash(uint64_t tag)
    {
        std::array<uint8_t, MDB_HASH_SIZE> h{};
        uint64_t x = tag ^ 0x9E3779B97F4A7C15ULL;
        for (std::size_t i = 0; i < h.size(); ++i) {
            // xorshift64*
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            const uint64_t y = x * 0x2545F4914F6CDD1DULL;
            h[i] = static_cast<uint8_t>((y & 0xFFu) ^ static_cast<uint8_t>(i));
        }
        return h;
    }

    inline std::vector<uint8_t> make_value_hash0(const std::array<uint8_t, MDB_HASH_SIZE>& h,
        const std::string& payload)
    {
        std::vector<uint8_t> v;
        v.reserve(MDB_HASH_SIZE + payload.size());
        v.insert(v.end(), h.begin(), h.end());
        v.insert(v.end(), payload.begin(), payload.end());
        return v;
    }

    inline std::vector<uint8_t> make_value_hash_at_offset(std::size_t offset,
        const std::array<uint8_t, MDB_HASH_SIZE>& h,
        const std::string& payload)
    {
        std::vector<uint8_t> v;
        v.reserve(offset + MDB_HASH_SIZE + payload.size());
        v.insert(v.end(), offset, 0xAB); // prefix bytes not included in hashsum
        v.insert(v.end(), h.begin(), h.end());
        v.insert(v.end(), payload.begin(), payload.end());
        return v;
    }


    inline std::vector<uint8_t> make_key_hash_at_offset(std::size_t offset,
        const std::string& order_prefix,
        const std::array<uint8_t, MDB_HASH_SIZE>& h,
        const std::string& suffix = std::string())
    {
        // Key layout:
        //   [0..offset)     : order_prefix (padded with 'X' up to offset)
        //   [offset..+16)   : hash bytes (included in hashsum)
        //   [..end)         : suffix (optional)
        std::vector<uint8_t> k;
        k.resize(offset, static_cast<uint8_t>('X'));
        const std::size_t n = std::min<std::size_t>(order_prefix.size(), offset);
        std::memcpy(k.data(), order_prefix.data(), n);
        k.insert(k.end(), h.begin(), h.end());
        k.insert(k.end(), suffix.begin(), suffix.end());
        return k;
    }

    inline bool bytes_equal(const std::string_view v, const std::vector<uint8_t>& b)
    {
        if (v.size() != b.size()) return false;
        return std::memcmp(v.data(), b.data(), b.size()) == 0;
    }
    inline std::array<uint8_t, MDB_HASH_SIZE> sum_hashes(const std::vector<std::array<uint8_t, MDB_HASH_SIZE>>& hashes)
    {
        std::array<uint8_t, MDB_HASH_SIZE> acc{};
        for (const auto& h : hashes) {
            ::mdb_hashsum_add(acc.data(), h.data());
        }
        return acc;
    }

    inline std::array<uint8_t, MDB_HASH_SIZE> sum_hash_regions(const std::vector<std::vector<uint8_t>>& values,
        std::size_t offset)
    {
        std::array<uint8_t, MDB_HASH_SIZE> acc{};
        for (const auto& v : values) {
            REQUIRE(v.size() >= offset + MDB_HASH_SIZE);
            ::mdb_hashsum_add(acc.data(), v.data() + offset);
        }
        return acc;
    }

    inline bool hash_equal(const uint8_t* a, const uint8_t* b)
    {
        for (std::size_t i = 0; i < MDB_HASH_SIZE; ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    struct temp_env_path {
        std::string path;

        temp_env_path()
        {
            const auto pid = static_cast<unsigned long>(::getpid());
            const auto now = static_cast<unsigned long long>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::ostringstream oss;
            oss << "/tmp/lmdbxx_agg_test_" << pid << "_" << now << ".mdb";
            path = oss.str();

            // Best-effort cleanup of leftovers (shouldn't exist).
            std::remove(path.c_str());
            const std::string lock = path + "-lock";
            std::remove(lock.c_str());
        }

        ~temp_env_path() noexcept
        {
            std::remove(path.c_str());
            const std::string lock = path + "-lock";
            std::remove(lock.c_str());
        }
    };

    struct fixture {
        temp_env_path tmp;
        lmdb::env env;

        fixture()
            : env(lmdb::env::create())
        {
            env.set_max_dbs(32)
                .set_mapsize(64u * 1024u * 1024u);

            unsigned int eflags = MDB_NOSUBDIR | MDB_NOLOCK | MDB_NOSYNC | MDB_NOMETASYNC;
            // eflags |= MDB_AGG_CHECK;
            env.open(tmp.path.c_str(), eflags, 0664);
        }
    };

    inline std::string key_k(unsigned i)
    {
        std::ostringstream oss;
        oss << "k" << std::setw(2) << std::setfill('0') << i;
        return oss.str();
    }

    // ------------------------------ Tests ------------------------------

    void test_plain_agg_features(fixture& fx)
    {
        const unsigned int db_flags =
            MDB_CREATE | MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM;

        lmdb::dbi db;

        std::vector<std::array<uint8_t, MDB_HASH_SIZE>> hashes;
        hashes.reserve(10);

        // Create + seed
        {
            auto wtxn = lmdb::txn::begin(fx.env);
            db = lmdb::dbi::open(wtxn, "plain", db_flags);
            db.set_hash_offset(wtxn, 0);

            for (unsigned i = 0; i < 10; ++i) {
                const auto kstr = key_k(i);
                std::string_view k{ kstr };

                const auto h = make_hash(static_cast<uint64_t>(0xA55AA55Au) ^ i);
                hashes.push_back(h);

                const auto vbytes = make_value_hash0(h, std::string("value/") + kstr);
                std::string_view v{ reinterpret_cast<const char*>(vbytes.data()), vbytes.size() };
                db.put(wtxn, k, v, 0);
            }
            wtxn.commit();
        }

        // Query + verify
        {
            auto rtxn = lmdb::txn::begin(fx.env, nullptr, MDB_RDONLY);

            // Schema flags
            const auto schema = db.agg_flags(rtxn);
            REQUIRE((schema & MDB_AGG_ENTRIES) != 0);
            REQUIRE((schema & MDB_AGG_KEYS) != 0);
            REQUIRE((schema & MDB_AGG_HASHSUM) != 0);

            // Totals
            const auto total = db.totals(rtxn);
            REQUIRE(total.has_entries());
            REQUIRE(total.has_keys());
            REQUIRE(total.has_hashsum());
            REQUIRE_EQ(total.mv_agg_entries, 10u);
            REQUIRE_EQ(total.mv_agg_keys, 10u);

            const auto expected_total = sum_hashes(hashes);
            REQUIRE(hash_equal(total.hashsum_data(), expected_total.data()));

            // Prefix: strictly less than k05 -> k00..k04
            std::string_view bnd{ "k05" };
            const auto p = db.prefix(rtxn, bnd, 0);
            REQUIRE_EQ(p.mv_agg_entries, 5u);
            REQUIRE_EQ(p.mv_agg_keys, 5u);
            {
                std::vector<std::array<uint8_t, MDB_HASH_SIZE>> sub(hashes.begin(), hashes.begin() + 5);
                const auto exp = sum_hashes(sub);
                REQUIRE(hash_equal(p.hashsum_data(), exp.data()));
            }

            // Prefix inclusive: include k05 -> 6
            const auto pi = db.prefix(rtxn, bnd, MDB_AGG_PREFIX_INCL);
            REQUIRE_EQ(pi.mv_agg_entries, 6u);
            REQUIRE_EQ(pi.mv_agg_keys, 6u);

            // Range: (k03..k07] default exclusive, then inclusive both
            std::string_view low{ "k03" };
            std::string_view high{ "k07" };

            const auto r_excl = db.range(rtxn, low, high, 0);
            // exclusive/exclusive => k04,k05,k06
            REQUIRE_EQ(r_excl.mv_agg_entries, 3u);
            REQUIRE_EQ(r_excl.mv_agg_keys, 3u);

            const unsigned range_incl = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;
            const auto r_incl = db.range(rtxn, low, high, range_incl);
            // inclusive/inclusive => k03..k07
            REQUIRE_EQ(r_incl.mv_agg_entries, 5u);
            REQUIRE_EQ(r_incl.mv_agg_keys, 5u);

            // Rank exact (entries weight): k04 => 4
            {
                std::string_view k{ "k04" };
                std::string_view d{}; // ignored for non-DUPSORT
                uint64_t rank = 0;
                const bool found = db.rank(rtxn, k, d, lmdb::agg_weight::entries, lmdb::agg_rank_mode::exact, rank);
                REQUIRE(found);
                REQUIRE_EQ(rank, 4u);
            }

            // Rank set-range: k04a => first >= k05, rank 5, key updated
            {
                std::string_view k{ "k04a" };
                std::string_view d{};
                uint64_t rank = 0;
                const bool found = db.rank(rtxn, k, d, lmdb::agg_weight::entries, lmdb::agg_rank_mode::set_range, rank);
                REQUIRE(found);
                REQUIRE_EQ(rank, 5u);
                REQUIRE_EQ(to_string_copy(k), std::string("k05"));
            }

            // Select: rank 7 => k07
            {
                std::string_view k{};
                std::string_view d{};
                const bool found = db.select(rtxn, lmdb::agg_weight::entries, 7, k, d);
                REQUIRE(found);
                REQUIRE_EQ(to_string_copy(k), std::string("k07"));
            }

            rtxn.abort();
        }
    }

    void test_dupsort_agg_features(fixture& fx)
    {
        const unsigned int db_flags =
            MDB_CREATE | MDB_DUPSORT | MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM;

        lmdb::dbi db;

        // Keep values so we can compute expected hash sums for some subranges.
        // Values are crafted so that duplicate ordering is determined by byte[0].
        std::vector<std::vector<uint8_t>> all_values_in_order; // record-order: a0,a1,a2,b0,c0,c1

        auto make_dup_value = [](char key_letter, uint8_t dup_index) {
            std::array<uint8_t, MDB_HASH_SIZE> h{};
            h.fill(0);
            h[0] = dup_index; // determines ordering within dup set
            h[1] = static_cast<uint8_t>(key_letter);
            return make_value_hash0(h, std::string("dup/") + key_letter + "/" + std::to_string(dup_index));
            };

        // Create + seed
        {
            auto wtxn = lmdb::txn::begin(fx.env);
            db = lmdb::dbi::open(wtxn, "dups", db_flags);
            db.set_hash_offset(wtxn, 0);

            const auto put_dup = [&](const char* kstr, uint8_t idx) {
                std::string_view k{ kstr };
                const auto vbytes = make_dup_value(kstr[0], idx);
                std::string_view v{ reinterpret_cast<const char*>(vbytes.data()), vbytes.size() };
                db.put(wtxn, k, v, 0);
                all_values_in_order.push_back(vbytes);
                };

            // a: 3 dups
            put_dup("a", 0);
            put_dup("a", 1);
            put_dup("a", 2);
            // b: 1 dup
            put_dup("b", 0);
            // c: 2 dups
            put_dup("c", 0);
            put_dup("c", 1);

            wtxn.commit();
        }

        // Verify totals + semantics
        {
            auto rtxn = lmdb::txn::begin(fx.env, nullptr, MDB_RDONLY);

            const auto total = db.totals(rtxn);
            REQUIRE_EQ(total.mv_agg_entries, 6u);
            REQUIRE_EQ(total.mv_agg_keys, 3u);

            // Key-only prefix < "b": all a duplicates
            {
                std::string_view kb{ "b" };
                const auto p = db.prefix(rtxn, kb, 0);
                REQUIRE_EQ(p.mv_agg_entries, 3u);
                REQUIRE_EQ(p.mv_agg_keys, 1u);

                const auto exp = sum_hash_regions({ all_values_in_order.begin(), all_values_in_order.begin() + 3 }, 0);
                REQUIRE(hash_equal(p.hashsum_data(), exp.data()));
            }

            // Record-order prefix <= ("b", b0): include a0,a1,a2,b0 => entries=4 keys=2
            {
                std::string_view kb{ "b" };
                // boundary data is b0
                const auto b0 = all_values_in_order[3];
                std::string_view db0{ reinterpret_cast<const char*>(b0.data()), b0.size() };
                const auto p = db.prefix(rtxn, kb, db0, MDB_AGG_PREFIX_INCL);
                REQUIRE_EQ(p.mv_agg_entries, 4u);
                REQUIRE_EQ(p.mv_agg_keys, 2u);
            }

            // Record-order range: [("a",a0), ("b",b0)] inclusive => a0,a1,a2,b0 => entries=4 keys=2
            {
                std::string_view ka{ "a" };
                const auto a0 = all_values_in_order[0];
                std::string_view da0{ reinterpret_cast<const char*>(a0.data()), a0.size() };
                std::string_view kb{ "b" };
                const auto b0 = all_values_in_order[3];
                std::string_view db0{ reinterpret_cast<const char*>(b0.data()), b0.size() };

                const unsigned range_incl = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;
                const auto r = db.range(rtxn, ka, da0, kb, db0, range_incl);
                REQUIRE_EQ(r.mv_agg_entries, 4u);
                REQUIRE_EQ(r.mv_agg_keys, 2u);
            }

            // Rank entry exact for (a,a1): rank=1, dup_index=1
            {
                std::string_view k{ "a" };
                const auto a1 = all_values_in_order[1];
                std::string_view d{ reinterpret_cast<const char*>(a1.data()), a1.size() };
                uint64_t rank = 0;
                uint64_t dup_index = 999;
                const bool found = db.rank(rtxn, k, d, lmdb::agg_weight::entries, lmdb::agg_rank_mode::exact, rank, &dup_index);
                REQUIRE(found);
                REQUIRE_EQ(rank, 1u);
                REQUIRE_EQ(dup_index, 1u);
            }

            // Rank keys exact for key "c": rank=2 (a=0,b=1,c=2)
            {
                std::string_view k{ "c" };
                uint64_t rank = 0;
                const bool found = db.rank(rtxn, k, lmdb::agg_rank_mode::exact, rank);
                REQUIRE(found);
                REQUIRE_EQ(rank, 2u);
            }

            // Select entry rank 5 => (c,c1), dup_index=1
            {
                std::string_view k{};
                std::string_view d{};
                uint64_t dup_index = 999;
                const bool found = db.select(rtxn, lmdb::agg_weight::entries, 5, k, d, &dup_index);
                REQUIRE(found);
                REQUIRE_EQ(to_string_copy(k), std::string("c"));
                REQUIRE_EQ(dup_index, 1u);
                REQUIRE_EQ(byte_at(d, 0), 1u); // our crafted ordering byte
            }

            // Select key rank 1 => "b", data is first dup (b0), dup_index=0
            {
                std::string_view k{};
                std::string_view d{};
                uint64_t dup_index = 999;
                const bool found = db.select(rtxn, lmdb::agg_weight::keys, 1, k, d, &dup_index);
                REQUIRE(found);
                REQUIRE_EQ(to_string_copy(k), std::string("b"));
                REQUIRE_EQ(dup_index, 0u);
                REQUIRE_EQ(byte_at(d, 0), 0u);
            }

            rtxn.abort();
        }
    }

    void test_hash_offset_nonzero(fixture& fx)
    {
        const unsigned int db_flags =
            MDB_CREATE | MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM;

        const std::size_t offset = 8;

        lmdb::dbi db;
        std::vector<std::vector<uint8_t>> values;
        values.reserve(4);

        // Create + seed
        {
            auto wtxn = lmdb::txn::begin(fx.env);
            db = lmdb::dbi::open(wtxn, "hashoff", db_flags);
            db.set_hash_offset(wtxn, static_cast<unsigned>(offset));

            for (unsigned i = 0; i < 4; ++i) {
                const std::string kstr = std::string("h") + std::to_string(i);
                std::string_view k{ kstr };

                const auto h = make_hash(0xC0FFEEULL ^ i);
                const auto vbytes = make_value_hash_at_offset(offset, h, std::string("payload/") + kstr);
                std::string_view v{ reinterpret_cast<const char*>(vbytes.data()), vbytes.size() };
                db.put(wtxn, k, v, 0);
                values.push_back(vbytes);
            }
            wtxn.commit();
        }

        // Verify totals hashsum uses offset bytes
        {
            auto rtxn = lmdb::txn::begin(fx.env, nullptr, MDB_RDONLY);
            const auto total = db.totals(rtxn);
            REQUIRE_EQ(total.mv_agg_entries, 4u);
            REQUIRE_EQ(total.mv_agg_keys, 4u);

            const auto exp = sum_hash_regions(values, offset);
            REQUIRE(hash_equal(total.hashsum_data(), exp.data()));
            rtxn.abort();
        }
    }



    void test_keyhash_plain_offset31(fixture& fx)
    {
        const unsigned int db_flags =
            MDB_CREATE | MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM | MDB_AGG_HASHSOURCE_FROM_KEY;

        const std::size_t offset = 31;

        lmdb::dbi db;
        std::vector<std::vector<uint8_t>> keys;
        std::vector<std::array<uint8_t, MDB_HASH_SIZE>> hashes;
        keys.reserve(10);
        hashes.reserve(10);

        // Create + seed: small values (smaller than offset+MDB_HASH_SIZE) to ensure value size is irrelevant
        // for key-sourced hashsum.
        {
            auto wtxn = lmdb::txn::begin(fx.env);
            db = lmdb::dbi::open(wtxn, "kh_plain31", db_flags);
            db.set_hash_offset(wtxn, static_cast<unsigned>(offset));

            for (unsigned i = 0; i < 10; ++i) {
                std::ostringstream koss;
                koss << "K" << std::setw(2) << std::setfill('0') << i << "/ORDER/";
                const auto h = make_hash(0xBEEFULL ^ i);
                const auto kbytes = make_key_hash_at_offset(offset, koss.str(), h, "/END");
                keys.push_back(kbytes);
                hashes.push_back(h);

                std::string_view k{ reinterpret_cast<const char*>(kbytes.data()), kbytes.size() };
                const char vsmall[] = "v";
                std::string_view v{ vsmall, sizeof(vsmall) - 1 };
                db.put(wtxn, k, v, 0);
            }
            wtxn.commit();
        }

        // Verify schema + aggregates.
        {
            auto rtxn = lmdb::txn::begin(fx.env, nullptr, MDB_RDONLY);

            const auto schema = db.agg_flags(rtxn);
            REQUIRE((schema & MDB_AGG_ENTRIES) != 0);
            REQUIRE((schema & MDB_AGG_KEYS) != 0);
            REQUIRE((schema & MDB_AGG_HASHSUM) != 0);
            REQUIRE((schema & MDB_AGG_HASHSOURCE_FROM_KEY) != 0);

            const auto total = db.totals(rtxn);
            REQUIRE_EQ(total.mv_agg_entries, 10u);
            REQUIRE_EQ(total.mv_agg_keys, 10u);

            const auto exp_total = sum_hashes(hashes);
            REQUIRE(hash_equal(total.hashsum_data(), exp_total.data()));

            // Prefix: < key for i=5 -> 5 items (K00..K04)
            {
                std::array<uint8_t, MDB_HASH_SIZE> hz{}; // boundary hash doesn't matter (ordering is in prefix bytes)
                std::ostringstream b;
                b << "K05/ORDER/";
                const auto bbytes = make_key_hash_at_offset(offset, b.str(), hz, "/END");
                std::string_view bound{ reinterpret_cast<const char*>(bbytes.data()), bbytes.size() };
                const auto p = db.prefix(rtxn, bound, 0);
                REQUIRE_EQ(p.mv_agg_entries, 5u);
                REQUIRE_EQ(p.mv_agg_keys, 5u);

                std::vector<std::array<uint8_t, MDB_HASH_SIZE>> sub(hashes.begin(), hashes.begin() + 5);
                const auto exp = sum_hashes(sub);
                REQUIRE(hash_equal(p.hashsum_data(), exp.data()));
            }

            // Range inclusive: [K03..K07] => 5 items
            {
                std::string_view low{ reinterpret_cast<const char*>(keys[3].data()), keys[3].size() };
                std::string_view high{ reinterpret_cast<const char*>(keys[7].data()), keys[7].size() };
                const unsigned range_incl = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;
                const auto r = db.range(rtxn, low, high, range_incl);
                REQUIRE_EQ(r.mv_agg_entries, 5u);
                REQUIRE_EQ(r.mv_agg_keys, 5u);

                std::vector<std::array<uint8_t, MDB_HASH_SIZE>> sub(hashes.begin() + 3, hashes.begin() + 8);
                const auto exp = sum_hashes(sub);
                REQUIRE(hash_equal(r.hashsum_data(), exp.data()));
            }

            // Rank exact for K04 => 4
            {
                std::string_view k{ reinterpret_cast<const char*>(keys[4].data()), keys[4].size() };
                std::string_view d{};
                uint64_t rank = 0;
                const bool found = db.rank(rtxn, k, d, lmdb::agg_weight::entries, lmdb::agg_rank_mode::exact, rank);
                REQUIRE(found);
                REQUIRE_EQ(rank, 4u);
            }

            // Select rank 7 => K07
            {
                std::string_view k{};
                std::string_view d{};
                const bool found = db.select(rtxn, lmdb::agg_weight::entries, 7, k, d);
                REQUIRE(found);
                REQUIRE(bytes_equal(k, keys[7]));
            }

            rtxn.abort();
        }
    }

    void test_keyhash_overwrite_current_no_delta_offset31(fixture& fx)
    {
        const unsigned int db_flags =
            MDB_CREATE | MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM | MDB_AGG_HASHSOURCE_FROM_KEY;

        const std::size_t offset = 31;

        auto wtxn = lmdb::txn::begin(fx.env);
        auto db = lmdb::dbi::open(wtxn, "kh_cur31", db_flags);
        db.set_hash_offset(wtxn, static_cast<unsigned>(offset));

        // Insert a single record (small value). Keep the key bytes stable in a vector.
        const auto h = make_hash(0x12345678ULL);
        const auto kbytes = make_key_hash_at_offset(offset, "K00/CURRENT/", h, "/END");
        std::string_view k{ reinterpret_cast<const char*>(kbytes.data()), kbytes.size() };
        const char v0[] = "small";
        std::string_view v{ v0, sizeof(v0) - 1 };
        db.put(wtxn, k, v, 0);

        const auto before = db.totals(wtxn);

        // Update current record with a much larger value via MDB_CURRENT, passing the key returned by cursor_get
        // (which may point into the leaf page and becomes unsafe if the page is modified).
        auto cur = lmdb::cursor::open(wtxn, db);
        std::string_view ck{}, cv{};
        REQUIRE(cur.get(ck, cv, MDB_FIRST));

        std::vector<uint8_t> big(3500, 0x5A);
        std::string_view vbig{ reinterpret_cast<const char*>(big.data()), big.size() };
        cur.put(ck, vbig, MDB_CURRENT);

        const auto after = db.totals(wtxn);

        REQUIRE_EQ(before.mv_agg_entries, after.mv_agg_entries);
        REQUIRE_EQ(before.mv_agg_keys, after.mv_agg_keys);

        // In key-hash mode, overwriting only the value must not change hashsum.
        REQUIRE(hash_equal(before.hashsum_data(), after.hashsum_data()));

        wtxn.commit();
    }

    void test_keyhash_dupsort_rejected(fixture& fx)
    {
        // Key-sourced hashsum is intentionally incompatible with DUPSORT.
        const unsigned int db_flags =
            MDB_CREATE | MDB_DUPSORT | MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM | MDB_AGG_HASHSOURCE_FROM_KEY;

        auto wtxn = lmdb::txn::begin(fx.env);

        bool threw = false;
        try {
            (void)lmdb::dbi::open(wtxn, "kh_dups_bad", db_flags);
        }
        catch (const lmdb::error& e) {
            threw = true;
            REQUIRE_EQ(e.code(), MDB_INCOMPATIBLE);
        }
        REQUIRE(threw);
        wtxn.abort();
    }


    void test_incompatible_dbi_throws(fixture& fx)
    {
        lmdb::dbi db;

        // Create a non-aggregate DBI
        {
            auto wtxn = lmdb::txn::begin(fx.env);
            db = lmdb::dbi::open(wtxn, "noagg", MDB_CREATE);
            std::string_view k{ "x" };
            std::string_view v{ "y" };
            db.put(wtxn, k, v, 0);
            wtxn.commit();
        }

        // totals() should throw MDB_INCOMPATIBLE
        {
            auto rtxn = lmdb::txn::begin(fx.env, nullptr, MDB_RDONLY);

            bool threw = false;
            try {
                (void)db.totals(rtxn);
            }
            catch (const lmdb::error& e) {
                // Expect MDB_INCOMPATIBLE from the aggregate call on a non-aggregate DBI.
                threw = true;
                REQUIRE_EQ(e.code(), MDB_INCOMPATIBLE);
            }
            REQUIRE(threw);

            rtxn.abort();
        }
    }

} // namespace test

int main()
{
    using test::fixture;

    struct tc {
        const char* name;
        std::function<void(fixture&)> fn;
    };

    const std::vector<tc> tests = {
      {"plain: totals/prefix/range/rank/select", test::test_plain_agg_features},
      {"dupsort: record-order + keys-vs-entries", test::test_dupsort_agg_features},
      {"hash offset: nonzero", test::test_hash_offset_nonzero},
      {"keyhash: plain (off=31) totals/prefix/range/rank/select", test::test_keyhash_plain_offset31},
      {"keyhash: overwrite + MDB_CURRENT no delta (off=31)", test::test_keyhash_overwrite_current_no_delta_offset31},
      {"keyhash: dupsort rejected", test::test_keyhash_dupsort_rejected},
      {"errors: incompatible totals()", test::test_incompatible_dbi_throws},
    };

    std::size_t passed = 0;
    std::size_t failed = 0;

    for (const auto& t : tests) {
        try {
            fixture fx;
            t.fn(fx);
            ++passed;
            std::cout << "[PASS] " << t.name << "\n";
        }
        catch (const test::failure& e) {
            ++failed;
            std::cout << "[FAIL] " << t.name << " - " << e.what() << "\n";
        }
        catch (const lmdb::error& e) {
            ++failed;
            std::cout << "[FAIL] " << t.name << " - lmdb::error: " << e.what() << "\n";
        }
        catch (const std::exception& e) {
            ++failed;
            std::cout << "[FAIL] " << t.name << " - std::exception: " << e.what() << "\n";
        }
        catch (...) {
            ++failed;
            std::cout << "[FAIL] " << t.name << " - unknown exception\n";
        }
    }

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed\n";
    return failed ? 1 : 0;
}
