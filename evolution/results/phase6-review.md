# 04 aggmaint v2: la manutenzione nell'unwind (24/09/2026)

Fase 6 del piano (`docs/plans/2026-09-23_piano_bottomup_v2.md`). Questo
documento riporta:
1. com'è fatta v2;
2. come è stata verificata contro v1;
3. i difetti trovati;
4. cosa resta.

Lo stadio è `lineage/04-aggmaint` (README con l'algoritmo), la serie
`patches/lineage/04-aggmaint`.

## 1. L'algoritmo

Una put o una delete cambia **un elemento**:
- un record di un DB piatto;
- un dupset intero nell'albero primario DUPSORT (conta come una chiave);
- un duplicato nel dup-tree.

Il suo contributo passa da `before` ad `after`, letti dal nodo al cursore
prima e dopo l'operazione. L'unwind di 02 sa già fin dove l'operazione ha
cambiato la struttura (`mt_unwind_prefix`). A operazione finita
`mdb_agg_settle()` percorre una volta il percorso finale del cursore:
- sotto il prefisso, ogni link riceve il fold della pagina figlia finale
  (una pagina, mai un sottoalbero);
- nel prefisso intatto, ogni link riceve Δ = after − before;
- i totali del DB ricevono Δ una volta.

I link che un passo strutturale crea o sposta fuori dal percorso li rende
esatti il passo stesso, quando le sue pagine sono finali:

| passo | link resi esatti |
|---|---|
| split | la pagina divisa e la nuova metà destra |
| move | i due fratelli |
| merge | la destinazione |
| nuovo sub-DB | i totali del record, dal fold della radice |

| serie | contenuto | righe di `mdb.c` |
|---|---|---:|
| V2M1 | algebra di v1 M1, più `mdb_aggval_equal` | +216 |
| V2M2 | fold locale di v1 M2, più `mdb_flat_leaf_item_agg` | +299 |
| V2M3 | settle, hook di put/delete, `mt_unwind_sub_prefix`, drop(0), oracolo | +222/−13 |
| V2M4 | SPLIT | +64/−4 |
| V2M5 | UNDERFLOW | +10 |
| V2M6 | DUPSORT: totali del nuovo sub-DB | +11 |
| V2M7 | contratto RESERVE, documentazione di `lmdb.h` | +10/−1 |
| V2Q1 | query di v1 (con C5), fold dei subpage inline | +5 |

In tutto `mdb.c` da 03 fa +836/−17 righe (v1 +1653/−37). La manutenzione vera
e propria (V2M3–V2M6) sono circa 300 righe, contro circa 1300 di v1 (M3–M6,
C1–C7).

### Il confronto con v1

v1 catturava un percorso prima dell'operazione (wrapper,
`MDB_agg_update_ctx`) e lo rigiocava dopo, con fasi di chiusura diverse per
DB piatti, DUPSORT e casi strutturali. C1, C4, C6 e C7 correggevano tutte un
percorso catturato che non corrispondeva più all'albero. In v2 quel percorso
non c'è:
- **C1**: la settle avviene prima della rilocazione del cursore;
- **C4**: la settle cammina sul percorso finale;
- **C6**: non c'è controllo d'identità della pagina;
- **C7**: ogni uscita della put dopo una modifica passa dalla settle.

I test di regressione di v1 girano invariati su v2. Solo M3 ha bisogno di
`-DAGG_V2=1`: in v1 il test controllava un divieto temporaneo agli split, che
V2M3 non ha.

## 2. Verifica

**Criterio d'uscita (`tests/run_battery.sh aggmaint`).** Il confronto usa
l'op stream con DB aggregati:
- oracolo d'integrità dopo ogni scrittura;
- hash di 1, 7, 32, 33 e 255 byte;
- semi 1–3;
- modalità normale e focus.

v1 e v2 danno tracce identiche. Le impronte di ogni snapshot sono identiche
**con i valori aggregati**, cioè prefissi dei branch, `md_keys` e
`md_hashsum`: gli aggregati coincidono bit per bit (i file differiscono nei
byte per l'ordine di allocazione).

Poi ciascuna implementazione apre lo snapshot intermedio e quello finale
scritti da ciascuna (`tests/04-aggmaint/X1_cross_open.c`). Su ogni file
esegue:
1. l'oracolo e i totali;
2. una transazione che cancella e riscrive record;
3. di nuovo oracolo e totali;
4. il commit.

Le quattro uscite coincidono.

**Il resto della batteria.** Con v1 accanto come stadio `aggmaint-v1`:
- test di milestone (`make verify-milestones`);
- `make test-aggmaint` (v2 e v1) e suite AELMDB;
- op stream in tutte le modalità (hazard e focus inclusi);
- long key con oracolo e impronte v2 = v1;
- confronto a tre AELMDB / v2 / v1 (`make compare`, firme identiche).

**ASan/UBSan.** Puliti su tutti questi carichi:
- op stream normale, focus e hazard (hash 33);
- long key;
- test di milestone e di query.

**Copertura.** gcov delle funzioni nuove sull'op stream aggregato e sui long
key. Sono state eseguite tutte le righe tranne:
- i ritorni d'errore;
- due rami difensivi della settle;
- i rami che l'op stream non usa: in `mdb_agg_put_check`, la sorgente chiave
  e il rifiuto di `MDB_RESERVE`; nella settle, lo schema senza KEYS né
  HASHSUM.

Questi ultimi li coprono i test di milestone (M3, M7) e il confronto.

**Benchmark (`tools/run_bench.sh`, indicativo).** Mediana di 3 ripetizioni,
in secondi, `MDB_NOSYNC`, una macchina a 2 core. I carichi sono: plain con
400k chiavi; DUPSORT con 100k chiavi × 4 duplicati.

| put | base | v2 off | v2 entries | v2 all | v1 entries | v1 all | AELMDB all |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain | 0,676 | 0,726 | 0,726 | 0,937 | 0,930 | 1,219 | 0,764 |
| DUPSORT | 0,339 | 0,393 | 0,436 | 0,625 | 0,814 | 1,133 | 0,557 |

Get, scan e delete restano vicini alla base in tutte le varianti. v2 con tutti
gli aggregati inserisce circa 1,3 volte più veloce di v1 sul DB piatto e 1,8
volte sul DUPSORT: non c'è la ricerca preliminare dei wrapper né il fold del
subpage in `mdb_xcursor_init1`. Rispetto alla base il costo è 1,39× (plain) e
1,84× (DUPSORT, hashsum dal valore). L'obiettivo della fase 7 è 1,3–1,5×.

## 3. Difetti trovati

- **Totali di un subpage inline (v2).** In v1 `mdb_xcursor_init1` faceva il
  fold del subpage nel `MDB_db` del sub-cursore a ogni lettura che entra in
  un dupset (M6). v2 non tocca il percorso di lettura: un subpage inline non
  ha totali propri. Il layer di query però li leggeva, e un limite oltre
  l'ultimo duplicato di un dupset inline dava keys e hashsum a zero.
  - Trovato dalla suite AELMDB (`dupsort round robin`). L'oracolo non poteva
    vederlo (l'albero era giusto), e nemmeno C5, che controlla solo le
    entries.
  - Correzione in V2Q1: il layer di query fa il fold del subpage quando gli
    servono i totali.
  - Regressione `Q2_dupset_bounds.c`: entries, keys e hashsum dei limiti
    appena fuori da un dupset, inline e sub-DB. Passa su v1 e fallisce su v2
    senza la correzione.
- **`make test-aggmaint` nascondeva i fallimenti.** Ogni test era in
  pipeline con `tail -n 1`, che dà sempre stato 0. Adesso il log va su file,
  e un fallimento lo stampa e ferma il target.
  - L'unico fallimento così nascosto era C7 con hash > 32 (su v1 come su v2):
    i valori del test erano più corti di una fetta di hash. Ora il test li
    dimensiona sulla fetta.
- **E3 in F1.** La regressione di E3 (riapertura di un MAIN DUPSORT non
  vuoto, query di schema su una txn bloccata) è ora in F1:
  - fallisce su 03 prima di E3;
  - passa su 03 e su entrambi i 04;
  - su v1 03, che precede E3 (v1 lo corregge in M7), è una riga KNOWN.

## 4. Cosa resta

- Fase 7: T3/T4, benchmark ripetuti su una macchina scarica (il DUPSORT con
  hashsum è sopra l'obiettivo), copertura su tutta la manutenzione.
- Il confronto condiviso (`tests/compare`) non ha ancora i limiti di dupset
  di Q2 né i confini assenti di C5: andrebbero aggiunti lì, così che valgano
  anche per AELMDB.
- Il ramo v1 resta vivo come riferimento, finché si vuole il confronto a tre.
