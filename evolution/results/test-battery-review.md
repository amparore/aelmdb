# Batteria di test di fase 0 — esito (2026-09-24)

Completa la fase 0 del piano (`docs/plans/2026-09-23_piano_bottomup_v2.md` §6–7):
la batteria che deve restare verde durante il refactor BU4–BU7, il rebase di
03 e aggmaint v2. Tutto è eseguibile con `make` e costruisce ogni stadio dai
sorgenti in `build/battery/`.

## 1. Cosa contiene

| Target | Contenuto | Oracolo | Tempo |
|---|---|---|---|
| `make test-lmdb` | `tests/lmdb/mtest*.c` (resi deterministici, `MTEST_SEED`) su lmdb, 01, 02, 03, 04, AELMDB × 3 seed; report `key_aliasing.c` | exit status; report identico a LMDB su ogni stadio della linea | ~20 s |
| `make test-e1` | `tests/01-base/E1_even_padding.c` | struttura attesa (1 pagina foglia) | <1 s |
| `make test-aelmdb-suite IMPL=aggmaint\|aelmdb` | `mtest_unit`, `mtest_unit_keyhash`, `mtest_adv` via `agg_test_compat.h`, con e senza oracolo di integrità | test stessi | ~1 min |
| `make test-large-keys` | `tests/struct/large_keys.c` (4 profili: near-max, half, 8 dup × 400 B, 32 dup + delete via cursore) | modello in memoria + scan completo + postcondizioni per operazione; hash del contenuto identico su tutti gli stadi; con `-DLK_AGG` integrità top-down e totali | ~75 s |
| `make test-struct` | `tests/struct/diff_ops.c`: flusso pseudo-casuale di put/del/cursor put/del/MULTIPLE/RESERVE/APPEND*/nested txn/drop con 3 cursori su DB plain, DUPSORT, DUPFIXED | traccia per operazione (rc, record, posizione di ogni cursore dopo ogni scrittura) + digest ordinato a checkpoint, **identica a quella di LMDB**; build aggregate con integrità a ogni checkpoint | ~85 s (50 seed) |
| `make test-smoke` | versioni ridotte dei precedenti | | ~3 min |
| `make test-battery` | tutto | | ~6 min |
| `make test-aggmaint` | ora include C5, C6, C7 | | |

Criterio strutturale (decisione del 23/09): fingerprint dell'albero
(`tools/lmdb_fingerprint.py`), non uguaglianza byte a byte. La batteria
confronta i fingerprint degli snapshot di fase (long-key) e di commit (op
stream) di 01 e 02: devono coincidere. Il confronto LMDB vs 01 è solo
informativo, perché il padding EVEN cambia il riempimento delle pagine.

Le query aggregate non sono più parte dei test strutturali: con `LK_AGG` o
`DS_AGG` si attiva il controllo top-down dell'intero albero
(`mdb_agg_check_integrity` / `mdb_dbg_check_agg_db`). Si può aggiungere il
confronto dei totali (`LK_AGG_TOTALS=1`). Le query restano coperte da Q1, C5,
dalla suite AELMDB e dal compare condiviso.

## 2. Risultati

Tutti i target sono verdi sulla linea 01–04. Le righe del ramo AELMDB
congelato sono riportate come KNOWN e non fanno fallire la batteria.

Sweep aggiuntivi eseguiti a mano con il flusso differenziale:

- 200 seed × 20 000 operazioni;
- 40 seed su 60 chiavi (churn dei dupset);
- 20 seed × 100 000 operazioni su 4 000 chiavi;
- build aggregata di 04: 300 seed, e 20 seed con integrità dopo *ogni*
  scrittura.

Tutte le tracce sono identiche a LMDB.

## 3. Bug trovati e corretti (collocati nell'albero delle patch)

| Id | Stadio | Difetto | Correzione | Regressione |
|---|---|---|---|---|
| **E1** | 01-base (eredita 02) | Il padding EVEN delle chiavi era applicato in `mdb_node_add` ma non dove lo spazio viene rilasciato o stimato: `mdb_node_del`, `mdb_page_split`, soglia sub-page→sub-DB, `mdb_page_list`. Ogni delete di un nodo con chiave e dato dispari perdeva 2 byte di pagina. La deriva finiva in split anticipati e in `MDB_PAGE_FULL` (transazione fallita) su put che LMDB accetta. DLMDB e 03 erano già corretti. | `EVEN(ksize)` nei 4 punti, come in DLMDB; incorporato in `01-base.patch` (è il completamento dell'import DLMDB); 02 e 03 rigenerati | `tests/01-base/E1_even_padding.c`, `make test-struct` (seed 3) |
| **E2** | 03-aggformat | Nel replace a pari dimensione di una chiave del sotto-albero dei duplicati, `NODEKEY(mp, leaf)` usava la variabile locale `mp`, che in quel ramo non è la pagina del nodo (non inizializzata: segfault). 04 lo correggeva solo in M6. | La correzione di M6 è spostata in 03; le patch M1–M6 sono rigenerate (lo snapshot M6 e quelli successivi sono invariati) | `make test-struct` (lo stadio 03 andava in segfault su ogni seed) |
| **C6** | 04-aggmaint | `mdb_cursor_put(MDB_CURRENT)` su DUPSORT/DUPFIXED aggregato con dupset in sub-page inline e pagina pulita al posizionamento → `MDB_INCOMPATIBLE` e transazione avvelenata. Il finish confrontava il pgno sintetico della sub-page con `md_root` stantio del sub-cursore. | Controllo di identità saltato per sub-page inline a pagina singola (nessun link padre) — patch 12 | `tests/04-aggmaint/C6_current_subpage.c` (milestone C6) |
| **C7** | 04-aggmaint | `MDB_APPENDDUP` fuori ordine su chiave con valore singolo. LMDB prima converte il valore in dupset (e può splittare la foglia), poi fallisce con `MDB_KEYEXIST`. Il wrapper usciva senza finish e i link dello split restavano non pubblicati (integrità: `MDB_CORRUPTED`). | Dopo quel `MDB_KEYEXIST` si esegue il finish DUPSORT (delta logico nullo) e si restituisce il risultato di LMDB — patch 13. Una prima versione rifiutava la put in anticipo, ma divergeva da LMDB in modo osservabile (vedi H3) ed è stata scartata. | `tests/04-aggmaint/C7_appenddup_order.c` (milestone C7) |

C5 (query `SET_RANGE` con `exactp`) era stato corretto nella sessione
precedente. `verify-lineage --milestones` ricostruisce tutti gli stadi dalle
patch con identità byte a byte ed esegue M1…M6, Q1, C4…C7 sugli snapshot
ricostruiti.

## 4. Confermati, non corretti

- **Aliasing della chiave** (`docs/findings/key_aliasing.md`): comportamento
  LMDB, mantenuto per decisione. Test corretto (copia della chiave, verifica
  dell'insieme esatto di chiavi) e comportamento documentato e fissato da
  `make test-lmdb`.
- **Hazard dei cursori LMDB 0.9.70** (`docs/findings/lmdb_cursor_hazards.md`):
  - H1: operazioni relative dopo un posizionamento assoluto fallito leggono
    stato stantio;
  - H2: `mdb_cursor_del` attraverso un cursore con `mx_db` stantio altera
    `ms_entries`;
  - H3: effetti collaterali osservabili di put fallite o nulle.

  Presenti in LMDB e riprodotti identici da tutti gli stadi. L'harness li evita
  per default; `DS_HAZARD=1` rimuove il re-seat. Il refactor, con la sua fase
  "settle" dei cursori, potrebbe correggere H2: sarebbe un cambiamento
  semantico deliberato da documentare.
- **Ramo AELMDB** (solo conferma, come richiesto):
  - C3 (`MDB_GET_BOTH` accetta match inesatto) emerge in ogni flusso con più
    duplicati. Con la patch di audit C3 applicata, AELMDB sull'API LMDB dà
    tracce identiche a LMDB.
  - **L1** (già noto dal compare long-key, `branches/aelmdb-initial/README.md`):
    con DB aggregati, delete e put producono `MDB_CORRUPTED`. Ora lo si
    riproduce anche senza query e con l'API LMDB: profilo long-key "half",
    dup lunghi (solo con l'oracolo di integrità compilato) e 36/50 flussi
    differenziali.
  - Differenze API minori: `MDB_FIRST_DUP`/`MDB_LAST_DUP` su DB non DUPSORT
    non restituiscono errore; `MDB_CURRENT` preserva la chiave aliasata.
  - `make test-aelmdb-suite IMPL=aelmdb` resta verde.

## 5. Strumenti aggiunti

- `tools/stage_src.sh`: sorgenti di uno stadio in una directory; include
  `aelmdb-c3`, cioè AELMDB con la patch di audit applicata.
- `tests/run_battery.sh`: driver della batteria.
- In `diff_ops`: `DS_STOP=N` salva lo stato persistente prima dell'operazione
  N, per ridurre un fallimento a un caso minimo (usato per C7/H3).
  `DS_SNAPSHOT_DIR` e `LK_SNAPSHOT_DIR` producono snapshot per il fingerprint.

## 6. Resta aperto (non bloccante per il refactor)

- Estendere il fingerprint al formato aggregato (prefisso nei branch, `MDB_db`
  più grande), per confronti strutturali 03/04 e poi v2 vs v1.
- Aggiungere al compare condiviso (T2) i confini assenti che hanno nascosto
  C5.
- ~~Target `coverage` (gcov) sul flusso differenziale e sui long-key~~: fatto
  dopo BU7 (`make coverage`, flusso focus `DS_FOCUS=1`), vedi
  `2026-09-24_copertura_BU7.md`.
- Test storici mancanti (BU1/BU2, `aggformat_*`) restano non recuperati. Il
  flusso differenziale ne copre in larga parte lo scopo per 01–03.
