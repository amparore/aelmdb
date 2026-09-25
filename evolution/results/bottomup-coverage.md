# Copertura di 02 dopo BU7 e il flusso focus (24/09/2026)

Il piano chiede di rimisurare la copertura all'uscita di BU7 (§8, fase 4), e
il §6.4 vuole che i rami non raggiunti diventino casi mirati nel generatore.
Questo documento riporta:
1. la prima misura;
2. il flusso focus costruito per chiudere le lacune;
3. i difetti che il flusso ha trovato;
4. la misura finale.

Strumento: `make coverage` (`tools/coverage.sh bottomup`), gcov su
`lineage/02-bottomup/mdb.c` compilato con `--coverage -O0`. Il rapporto con
le righe mai eseguite di ogni funzione è in `build/coverage/bottomup/report.txt`.

## 1. Prima misura

Carichi:
- flusso differenziale, semi 1–10, modalità normale e hazard;
- chiavi lunghe, cinque profili;
- `mtest`…`mtest6`, `key_aliasing`, E1, B1.

Risultato: **88,8% delle righe e 76,9% dei rami** (44 funzioni di put,
delete, rebalance, split, primitive locali e unwind).

Oltre ai percorsi di errore restavano scoperti questi rami strutturali:
- REKEY con risalita, cioè un `MDB_CURRENT` su un duplicato all'indice 0 di
  una foglia interna di un sub-DB;
- i fix-up di altri cursori durante il collasso della radice, un albero
  svuotato e un move;
- i move di nodi branch e il collasso della radice negli alberi di duplicati
  profondi, anche LEAF2;
- `MDB_INTEGERDUP`.

La misura su BU6 aveva mostrato codice morto in `mdb_put_sub`, che BU7 ha
tolto.

## 2. Il flusso focus (`DS_FOCUS=1`)

È una modalità di `tests/struct/diff_ops.c`. Il flusso di default resta
identico byte per byte. Rispetto al default, il flusso focus:
- usa poche chiavi (48, oppure 6 in una variante);
- alterna fasi di crescita e di contrazione;
- a ogni fase fa una **crescita di massa** di un dupset, da qualche centinaio
  a qualche migliaio di valori (MULTIPLE per DUPFIXED);
- poco dopo **contrae** lo stesso dupset con delete casuali
  (`GET_BOTH_RANGE` + `mdb_cursor_del`) o consecutive;
- usa valori DUPFIXED da 256 byte, così gli alberi di duplicati arrivano a
  profondità 3–4;
- aggiunge un quarto DB `DUPSORT|DUPFIXED|INTEGERDUP`;
- tiene i cursori soprattutto sullo stesso DB;
- fa spesso `MDB_CURRENT` dopo un riposizionamento su un duplicato qualsiasi;
- rende rare le delete dell'intero dupset.

`make test-struct` lo esegue su 20 semi nelle varianti:
- normale;
- 6 chiavi;
- API portabile;
- DB aggregati, con l'oracolo di integrità di aggmaint;
- hazard, con 01-base come riferimento.

`make test-unwind` lo esegue con gli oracoli di unwind. Le impronte 01 = 02
sono confrontate anche sui suoi snapshot.

## 3. Difetti trovati

**In 02 (BU7).** L'asserzione aggiunta in `mdb_put_sub` è scattata con
`MDB_MULTIPLE`. Gli elementi successivi a quello che ha costruito un sub-DB
vedevano ancora `ps_sub_root`: in LMDB era solo un suggerimento di cache per
la pagina radice. BU7 ora azzera lo stato del nuovo contenitore dopo quell'elemento. Senza
l'asserzione il comportamento era corretto, ma l'invariante dichiarato no.
Con la stessa misura si è visto che `ps_do_sub` era morto dopo BU6, e BU7 lo
ha tolto.

**In LMDB 0.9.70, ereditati da 01-base**
(`docs/findings/lmdb_cursor_hazards.md`). In modalità hazard 01-base
corrompeva `md_entries` e andava in abort, e 02, 03 e 04 lo seguivano con
tracce identiche. Le correzioni vanno quindi in 01-base: B1c–B1f, propagate
lungo tutta la catena.

| | pericolo | correzione |
|---|---|---|
| H2c | `MDB_CURRENT` con copia del dupset stantia: la put annidata riscrive la copia nel nodo | B1c: sync prima di `MDB_CURRENT` |
| H4 | la delete del cursore lo sposta su un sub-DB con un sub-cursore di un'altra chiave | B1d: sub-cursore azzerato con il nodo |
| H5 | `C_EOF` rimane dopo l'inserimento di una chiave oltre l'ultima | B1e |
| H6 | `FIRST_DUP`/`LAST_DUP` ripartono da una radice copiata da un'altra scrittura | B1f: sync del record a ogni get |

Il test di regressione è `tests/01-base/B1c_cursor_hazards.c`: fallisce su
LMDB e su 01 prima di B1c, e passa su 01, 02, 03 e 04. Nelle modalità con
riferimento LMDB l'harness evita questi percorsi con re-seat mirati, e le
tracce restano identiche a LMDB.

## 4. Misura finale

Ai carichi della prima misura si aggiungono il flusso focus (normale e hazard,
più la variante a 6 chiavi) e B1c.

| | eseguite | totale | % |
|---|---:|---:|---:|
| righe | 1407 | 1534 | **91,7** |
| rami presi | 799 | 962 | **83,1** |

Ora sono al 100% delle righe, tra le altre:
- `mdb_level_rekey_up` (compresa la risalita), `mdb_put_sub`,
  `mdb_put_dup_container`, `mdb_xcursor_init_container`;
- `mdb_split_fix_cursors`, `mdb_del0_node`, tutti i passi di discesa.

Altre coperture:

| funzione | righe |
|---|---:|
| `mdb_page_split_local` | 97% |
| `mdb_rebalance_root` | 94% |
| `mdb_node_move_local` | 91% (con i rami LEAF2 dei nodi branch) |

Righe ancora non eseguite:

- **Percorsi di errore** (quasi tutte):
  - `ENOMEM`, `mdb_page_get` fallita;
  - `EACCES` / `MDB_BAD_TXN` / `MDB_BAD_VALSIZE`;
  - `MDB_INCOMPATIBLE` per uso scorretto dell'API;
  - lo split locale fallito (`MDB_UP_PUSH_STOP`);
  - `mdb_txn_mark_error` nei passi di discesa.

  Servirebbe fault injection. Non bloccante.
- **Arena dei frame di `mdb_unwind`**: la coprono il build
  `-DMDB_UNWIND_FAST=2` della suite `unwind` e un giro con ASan.
- **Spill**: `mdb_page_unspill` in `mdb_put_current` richiede transazioni
  con più pagine sporche della dirty list.
- **Irraggiungibile in pratica**:
  - la ri-chiave del nodo 0 del genitore in `mdb_rebalance_apply_merge` (il
    sorgente di un merge non è mai il primo figlio);
  - la nuova chiave DUPSORT con `LEAFSIZE > me_nodemax` (pagine da 4 KB,
    `MDB_MAXKEYSIZE` 511);
  - il ramo `MDB_PROBLEM` di `mdb_node_move_local`, che è un invariante;
  - un DB `DUPFIXED` senza `DUPSORT` creato vuoto.

## Nota

UBSan segnala load disallineati in `mdb_cmp_long` per `INTEGERDUP` con
valori `size_t`. È il comportamento di LMDB 0.9.70, identico in tutti gli
stadi e innocuo su x86 (vedi il documento dei pericoli).
