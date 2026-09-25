# AELMDB baseline vs. aggmaint/Q1 — confronto normalizzato

## 1. Obiettivo

Questo confronto tratta `aggmaint` come una **baseline parallela e rivista** di AELMDB, non come una patch da integrare immediatamente in AELMDB.

Le due linee confrontate sono:

- **AELMDB baseline iniziale**: `mdb(2).c`, `lmdb(1).h`, `midl(1).c`, `midl(1).h`;
- **aggmaint/Q1**: `aggmaint_mdb.c`, `aggmaint_lmdb.h`, `mdb_agg_query.c`, `mdb_agg_debug.c`.

Le window query sono deliberatamente escluse dalla superficie normalizzata del confronto. Restano nel baseline AELMDB come funzionalità aggiuntiva, ma non sono usate come criterio per valutare il core.

La matrice concettuale è quindi:

| Area | AELMDB baseline | aggmaint/Q1 | Confronto |
|---|---|---|---|
| S — substrate strutturale | mutation flow LMDB adattato | bottom-up local-result | sì |
| F — formato | aggregate branch format A333 | aggregate format A335 | sì |
| M — maintenance | refresh/recompute + postcheck/repair | delta + exact structural publication | sì |
| Q — query base | totals/prefix/range/rank/select/seek | stesso API, file separato | sì |
| W — window query | presente | assente | **escluso** |
| D — diagnostica | integrity + print nel core | oracle separato, niente print | sì |

---

## 2. Compatibilità dell'API query

Il risultato più semplice e utile è che **non serve alcun compatibility shim per il test condiviso**.

Lo stesso sorgente C usa direttamente, su entrambe le implementazioni:

- `mdb_agg_info()`;
- `mdb_set_hash_offset()` / `mdb_get_hash_offset()`;
- `mdb_agg_totals()`;
- `mdb_agg_prefix()`;
- `mdb_agg_range()`;
- `mdb_agg_rank()`;
- `mdb_agg_select()`;
- `mdb_agg_cursor_seek_rank()`;
- le normali API LMDB per popolare, mutare e scandire il DB.

La sola differenza di build rilevante per il confronto comune è `MDB_HASH_SIZE`: l'header AELMDB baseline richiede ancora un multiplo di 8, mentre aggmaint supporta qualunque valore in `[1,256]`. La matrice condivisa usa quindi `8, 32, 64, 256`.

---

## 3. Test condiviso

È stato creato `tests/compare/agg_compare_shared.c`. È **lo stesso identico sorgente** compilato contro AELMDB e contro aggmaint/Q1.

L'oracle non usa internals aggregate: costruisce lo stato atteso tramite scansione ordinaria LMDB e calcola indipendentemente:

- numero di record;
- numero di chiavi distinte;
- hashsum modulare;
- prefix e range;
- rank e select;
- posizionamento di un cursor per entry-rank.

Il test verifica anche boundary/error semantics comuni:

- `select(rank == size) -> MDB_NOTFOUND`;
- `cursor_seek_rank(rank == size) -> MDB_NOTFOUND`;
- `rank(EXACT)` su chiave mancante -> `MDB_NOTFOUND`;
- range invertita -> aggregate nullo;
- `mdb_agg_info()` e `mdb_get_hash_offset()`.

### 3.1 Modalità `query`

Costruisce tre DB sufficientemente grandi da creare branch page e rappresentazioni DUPSORT diverse:

1. plain, value-source HASHSUM, offset `-1`;
2. DUPSORT, value-source HASHSUM;
3. plain, key-source HASHSUM.

Dopo commit/reopen confronta tutte le query con l'oracle.

### 3.2 Modalità `growth`

Dopo il primo ciclo:

- sostituisce valori plain;
- estende il bordo destro del tree;
- aumenta dupset esistenti;
- crea nuovi primary key DUPSORT;
- estende il DB key-source.

Verifica le query sia nella write transaction sia dopo commit/reopen. Questa modalità stressa in particolare split, DUPSORT growth e maintenance bottom-up senza delete/rebalance.

### 3.3 Modalità `stress`

Aggiunge alla modalità precedente:

- delete sparse plain;
- delete contigue sufficienti a cambiare il posizionamento del cursor e a provocare rebalance;
- delete di interi dupset;
- delete di singoli duplicate;
- successiva ricrescita.

Questa modalità è intenzionalmente più aggressiva sul maintenance core.

---

## 4. Risultati comportamentali

### 4.1 Query-only

| HASH | AELMDB | aggmaint/Q1 | signature uguale |
|---:|---|---|---|
| 8 | PASS `79ec10a97f1ca5ae` | PASS `79ec10a97f1ca5ae` | sì |
| 32 | PASS `030f74a873810842` | PASS `030f74a873810842` | sì |
| 64 | PASS `d33a6af3e6f345e5` | PASS `d33a6af3e6f345e5` | sì |
| 256 | PASS `b582e800f46a5cbc` | PASS `b582e800f46a5cbc` | sì |

**Conclusione Q:** sulle query base e sullo stato prodotto dal caricamento iniziale, le due implementazioni sono comportamentalmente equivalenti per tutti i width comuni provati. Non solo passano lo stesso oracle: producono la stessa firma deterministica.

### 4.2 Growth/split/DUPSORT growth

| HASH | AELMDB | aggmaint/Q1 | signature uguale |
|---:|---|---|---|
| 8 | PASS `995b9a2a97eb1866` | PASS `995b9a2a97eb1866` | sì |
| 32 | PASS `d89abf80abee3e62` | PASS `d89abf80abee3e62` | sì |
| 64 | PASS `efc8a65aabff70c1` | PASS `efc8a65aabff70c1` | sì |
| 256 | PASS `8c111077b7fb8db4` | PASS `8c111077b7fb8db4` | sì |

**Conclusione S/M sui growth path:** anche dopo mutation, split e crescita DUPSORT, entrambe arrivano allo stesso stato logico e tutte le query base concordano con l'oracle.

### 4.3 Delete/rebalance stress

Dopo la correzione C1, anche il workload più aggressivo passa su entrambe le implementazioni e produce la stessa firma deterministica:

| HASH | AELMDB | aggmaint/Q1 | signature uguale |
|---:|---|---|---|
| 8 | PASS `694c47f7becd42ce` | PASS `694c47f7becd42ce` | sì |
| 32 | PASS `d2e4fc9b303eddba` | PASS `d2e4fc9b303eddba` | sì |
| 64 | PASS `d2480569e2c81d0d` | PASS `d2480569e2c81d0d` | sì |
| 256 | PASS `db9070a05945d038` | PASS `db9070a05945d038` | sì |

Questa modalità attraversa plain delete, cursor relocation, redistribution/merge, DUPSORT delete e successiva ricrescita. Il confronto condiviso raggiunge quindi ora anche l'intera parte delete/rebalance che prima era bloccata dal finding C1.

---

## 5. Finding C1 — investigazione e correzione

### 5.1 Causa

Il failure non era una modifica strutturale mancata. Il caso minimo è una delete dell'ultimo record di una leaf che, dopo la rimozione, resta sopra la soglia di fill e non richiede rebalance.

LMDB può legittimamente riposizionare il cursor sul sibling successivo. Il pre-state path che ha ricevuto il logical delta resta però quello della leaf originale. Il vecchio `mdb_agg_plain_finish()` assumeva invece che, in un'operazione non strutturale, gli `mc_ki[]` finali dovessero coincidere con quelli catturati prima della delete.

Quindi venivano confusi due concetti diversi:

```text
cursor relocation       !=      structural mutation
```

Nel caso strumentato il parent index passava da 11 a 12 mentre `structuralp == 0`.

### 5.2 Perché non basta indebolire il check

Rimuovere semplicemente il confronto sugli indici avrebbe nascosto il problema: dopo la relocation, usare il path corrente del cursor per pubblicare il delta avrebbe potuto aggiornare il sibling sbagliato.

Serve invece conservare l'identità del **pre-state writable path** sul quale è avvenuta la mutazione.

### 5.3 Correzione

La delete ora ha due momenti distinti:

1. `mdb_agg_*_del_prepare()` cattura la contribuzione logica `before` e valida il delta, come prima;
2. `_mdb_cursor_del_raw()` esegue `mdb_cursor_touch()`; subito dopo il touch, ma prima di cancellare il record, cattura i `pgno` del path writable.

Questo timing è importante: prima del touch i pgno possono cambiare per copy-on-write; dopo il touch sono stabili per una delete non strutturale.

Il path stabile è memorizzato in un array locale del wrapper delete e il context mantiene soltanto un puntatore temporaneo ad esso. Quindi il fix non aumenta la dimensione permanente dei context put/dupsort.

Nel finish non strutturale:

- il root viene validato contro il pgno catturato;
- ogni parent page viene rifetchata tramite il pgno stabile;
- il link viene identificato tramite il `ki` pre-state;
- viene verificato che punti al child pgno catturato;
- il logical delta viene applicato a quel link, indipendentemente dalla posizione finale del cursor.

Lo stesso meccanismo è usato anche dal finish del primary DUPSORT, evitando la stessa classe di errore su delete che possono riposizionare il primary cursor.

Le operazioni strutturali continuano invece a usare il flag esplicito prodotto dal rebalance engine e la pubblicazione bottom-up esatta M5; il path snapshot non sostituisce né indebolisce quella semantica.

### 5.4 Regression test mirato

`test_aggmaint_m7.c` contiene ora un test C1 dedicato che:

- costruisce un tree multi-page;
- committa e apre una nuova write transaction, così il path deve passare attraverso COW;
- sceglie programmaticamente una leaf non terminale il cui fill rimarrà sopra `FILL_THRESHOLD` dopo la delete;
- cancella esattamente l'ultimo record della leaf;
- verifica che il cursor sia stato realmente spostato su un'altra leaf;
- verifica l'intero tree tramite l'oracle ricorsivo M6/M7;
- committa, riapre e verifica di nuovo.

Il test fallisce sulla versione pre-C1 esattamente nel `mdb_cursor_del()` e passa dopo la correzione.

### 5.5 Validazione post-fix

- M7: PASS per `MDB_HASH_SIZE = 1, 7, 31, 32, 33, 255, 256`;
- Q1: PASS sulla stessa matrice;
- ASAN+UBSAN + integrity oracle: PASS per 32 e 33;
- shared stress AELMDB/aggmaint: PASS con firma identica per 8, 32, 64 e 256.

C1 è quindi **chiuso** e la riga M del confronto shared non è più incompleta.

---

## 6. Finding C2: accesso disallineato in AELMDB baseline

La build `ASAN+UBSAN`, `MDB_HASH_SIZE=32`, del test query condiviso produce invece un finding nella baseline AELMDB.

Nel refresh di un named DB (`mdb(2).c`, circa linee 14471–14490) AELMDB fa:

```c
MDB_db *fresh = (MDB_db *)data.mv_data;
...
txn->mt_dbs[dbi] = *fresh;
```

`NODEDATA()` non garantisce l'allineamento richiesto da `MDB_db`; UBSAN segnala quindi accessi disallineati a `fresh->md_depth`, `fresh->md_root` e alla copia finale.

aggmaint/Q1 non presenta questo errore nel medesimo test sanitizer, perché i descriptor persistenti vengono letti tramite copia in un `MDB_db` correttamente allineato.

Questo è un esempio utile del valore del confronto parallelo: AELMDB supera il test funzionale stress, ma conserva un problema di UB che il ramo aggmaint ha già eliminato.

---

## 7. Confronto del query layer

### AELMDB baseline

Il query code base è incorporato direttamente in `mdb.c`. Le funzioni pubbliche sono equivalenti a Q1, ma la regione contiene anche:

- duplicazione della semantica leaf/DUPSORT;
- helper per partial dup prefixes;
- rank-space prefix machinery usata dalle window query;
- `mdb_agg_postcheck_root_totals()`, che in caso di mismatch può chiamare `mdb_agg_root_refresh_all_links()` e quindi **riparare** aggregate mentre si trova nella stessa regione logica delle query.

Per questo la regione non è puramente read-only dal punto di vista architetturale.

### aggmaint/Q1

`mdb_agg_query.c` è 939 linee ed è incluso testualmente da `aggmaint_mdb.c`.

Non contiene riferimenti a:

- `mdb_node_set_*`;
- `mdb_page_touch()`;
- `mdb_page_split()`;
- `mdb_rebalance()`;
- repair/postcheck helpers;
- dirty publication.

Riusa invece le primitive semantiche stabilizzate dal maintenance (`MDB_aggval`, contribution dei leaf item, branch prefix access). Le query diventano quindi consumer read-only del formato mantenuto.

### Dimensione normalizzata della query

Una stima per dependency/range del baseline AELMDB dà circa **1475 linee** per la query base, escludendo la machinery esclusivamente window. Q1 è **939 linee**.

La riduzione non va interpretata come sola compressione sintattica: una parte importante deriva dal fatto che Q1 riusa la semantica leaf già definita dal maintenance invece di reimplementarla.

---

## 8. Superficie sorgente

Linee fisiche attuali:

| File | AELMDB baseline | aggmaint/Q1 |
|---|---:|---:|
| main `mdb.c` | 18,250 | 13,849 |
| public header | 2,219 | 1,787 |
| query esterne | — | 939 |
| debug oracle esterno | — | 148 |
| **totale** | **20,469** | **16,723** |

Le window query AELMDB occupano approssimativamente 751 linee di implementazione più 66 linee di API/header. Sottraendole per un confronto più omogeneo, AELMDB resta attorno a **19,652 linee**, contro **16,723** di aggmaint/Q1.

Questo dato non è da solo un criterio di qualità, ma conferma che la separazione `core/query/debug` riduce significativamente la superficie del `mdb.c` da confrontare.

Rispetto all'LMDB originale presente nel progetto:

| Variante | `mdb.c` diff raw vs originale |
|---|---:|
| AELMDB baseline | `+14468 / -7688` |
| aggmaint/Q1 main file | `+3025 / -716` |

Il diff raw AELMDB è molto rumoroso e non misura soltanto la funzionalità aggregate; proprio per questo la matrice per concern è più informativa del numero totale di linee modificate.

---

## 9. Retrospettiva incrementale aggmaint

| Step | + | - | net | churn | Ruolo principale |
|---|---:|---:|---:|---:|---|
| M1 | 202 | 0 | +202 | 202 | algebra aggregate |
| M2 | 263 | 0 | +263 | 263 | exact local-page aggregate |
| M3 | 391 | 2 | +389 | 393 | logical delta plain |
| M4 | 214 | 93 | +121 | 307 | split publication |
| M5 | 133 | 39 | +94 | 172 | move/merge/root |
| M6 | 459 | 50 | +409 | 509 | DUPSORT boundary |
| M7 | 216 | 21 | +195 | 237 | integration/audit/debug separation |
| Q1 | 1260 | 0 | +1260 | 1260 | query API + external query module |

La forma della serie è coerente con l'architettura:

- M1/M2 sono additive foundations;
- M3 introduce il meccanismo logical-delta;
- M4/M5 hanno net ridotto perché sfruttano il substrate bottom-up;
- M6 è il vero salto semantico per DUPSORT;
- M7 è relativamente piccolo nel core e sposta diagnostica fuori;
- Q1 è grande come patch perché aggiunge un modulo query intero e l'API pubblica, non perché aumenti il maintenance core.

Il finding C1 mostra però anche il limite di una lettura puramente quantitativa: M5 può essere piccolo e architetturalmente corretto, ma un guard nel wrapper M3/M5 può comunque lasciare un caso operativo scoperto.

---

## 10. Matrice qualitativa S/F/M/Q/D

| Concern | AELMDB baseline | aggmaint/Q1 | Stato comparativo |
|---|---|---|---|
| **S: split** | mutazione e successivo refresh dei link | `split_local` produce left/right finali, poi publish | aggmaint più esplicito |
| **S: move/merge** | update con ricomputazioni/refresh | result post-state + publication bottom-up | aggmaint più esplicito |
| **S: cursor relocation** | tollerato dal maintenance corrente | pre-state writable path catturato dopo COW | **C1 chiuso** |
| **F: branch prefix** | aggregate persistente su branch | stesso concetto, format auditato | semanticamente allineati |
| **F: format tag** | A333 | A335 | differenza intenzionale |
| **F: HASH_SIZE** | multipli di 8 | 1..256 | aggmaint più generale |
| **M: plain non-structural** | maintenance integrato + fallback | one logical change O(height) | growth equivalenti |
| **M: split** | refresh/recompute | exact post-state publication | growth equivalenti |
| **M: delete/rebalance** | stress condiviso passa | stress condiviso passa dopo C1 | firme identiche |
| **M: DUPSORT growth** | passa | passa | firme identiche |
| **M: DUPSORT delete** | short stress passa; C3 può far cancellare il successore quando il duplicate richiesto è assente | short/long H=32 passano | **finding C3 AELMDB** |
| **Q: totals** | presente | presente | equivalente nel test |
| **Q: prefix** | presente | presente | equivalente nel test |
| **Q: range** | presente | presente | equivalente nel test |
| **Q: rank/select** | presente | presente | equivalente nel test |
| **Q: cursor seek** | presente | presente | equivalente nel test |
| **Q: mutation side effects** | regione contiene postcheck/repair | query module read-only | aggmaint più separato |
| **W: window** | presente | esclusa | fuori scope |
| **D: integrity** | inline e diffusa | file esterno | aggmaint più confinato |
| **D: print tracing** | presente | eliminato | aggmaint più pulito |
| **X: alignment safety** | UB nel named-DB refresh | copia allineata | **finding C2 AELMDB** |

---

## 11. Stato del confronto

Il confronto ha già prodotto due risultati forti e complementari:

1. **compatibilità comportamentale molto alta** sul subset comune: query e growth path producono firme identiche su tutti i width comuni provati;
2. il test condiviso trova problemi reali su entrambi i lati:
   - aggmaint: false structural/path mismatch durante delete cursor relocation;
   - AELMDB: accesso disallineato a `MDB_db` nel refresh dei named DB.

Per questo non è ancora utile formulare un giudizio finale “AELMDB vs aggmaint”. Il passo corretto è:

1. correggere C1 in aggmaint senza indebolire la semantica bottom-up;
2. rieseguire **lo stesso identico test condiviso**;
3. verificare che il `stress` arrivi anche alle delete DUPSORT;
4. solo allora chiudere la matrice M su delete/rebalance e DUPSORT delete;
5. mantenere C2 come issue separata della baseline AELMDB, non come motivo per alterare il workload.

Questo mantiene il confronto simmetrico e rende il test condiviso un vero regression contract tra le due linee.


---

## 12. Extended shared long stress

Dopo la chiusura di C1, il test condiviso è stato esteso con una modalità
`long` deterministica.  La modalità breve `query/growth/stress` rimane invariata
e continua a produrre firme identiche fra AELMDB e aggmaint sul subset comune.

La nuova modalità aggiunge decine di migliaia di mutazioni attraverso molte
transazioni, valori plain di dimensione variabile e overflow, churn DUPSORT,
nested transaction commit/abort, top-level abort, close/reopen periodici e
verifica delle query sia nel writer sia dopo riapertura.  In aggiunta alle
query aggregate, usa fingerprint indipendenti ottenuti mediante scansione LMDB
ordinaria per verificare:

- isolamento di un child abort;
- isolamento di un top-level abort;
- persistenza esatta dello stato dopo commit/reopen.

Per le delete il test controlla inoltre il pre-state tramite API LMDB ordinaria
e verifica il post-state immediatamente.  Questa proprietà è importante perché
un oracle costruito soltanto scandendo il tree *dopo* una mutazione non può
distinguere un maintenance aggregate corretto da una mutazione logica già
sbagliata.

Con `MDB_HASH_SIZE=32`, seed `0x6d64626167676c31`, 80 round e 400 operazioni per
round, aggmaint completa il workload. L'analisi di C3 ha mostrato che il primo
segnale al round 0, operazione 118, era un **false positive dell'oracle**:
la delete del duplicate esistente era corretta, ma `MDB_GET_BOTH` sulla coppia
appena rimossa ritornava erroneamente successo posizionandosi sul duplicate
successivo.

L'oracle è stato quindi reso indipendente da `MDB_GET_BOTH`, scandendo il solo
dup-run. Con questo controllo corretto emerge la conseguenza mutativa reale al
round 0, operazione 194: quando il duplicate richiesto è assente ma ne esiste uno
maggiore, AELMDB `mdb_del(key,data)` può ritornare successo e cancellare il
successore.

La causa è una modifica AELMDB al contratto interno di `mdb_cursor_set()` per
supportare `MDB_SET_RANGE` + `exactp` nelle query aggregate. Il percorso storico
`MDB_GET_BOTH`, che usa ricorsivamente `MDB_SET_RANGE`, non verifica però il bit
`ex2` e accetta quindi un lower-bound inexact come exact match. La correzione
minima consiste nel controllare esplicitamente `!ex2` nel wrapper
`MDB_GET_BOTH`; non serve né conviene revertire il nuovo comportamento di
`MDB_SET_RANGE`.

Il patch isolato C3 fa passare il reproducer minimale e il long H=32 completo;
patched AELMDB e aggmaint producono la stessa firma `585f21a81eda71b8`. La
baseline AELMDB resta congelata; il patch è materiale di audit, non viene
incorporato nella baseline comparativa.

Questo è classificato come **C3**, distinto da C2 (UB/alignment del named-DB
refresh). I dettagli e il reproducer sono in `docs/compare/C3_get_both_audit.md` e
`tests/compare/c3_get_both.c`.

---

## 13. Large-key structural stress

Il confronto è stato esteso con un secondo asse di stress, deliberatamente
ortogonale al semplice aumento del numero di operazioni.  Le chiavi primarie
sono portate vicino a `mdb_env_get_maxkeysize()` così da ridurre insieme la
capacità delle leaf e il fanout dei branch; le separator key grandi propagano
quindi la pressione strutturale a tutti i livelli del B+tree.

Sono mantenuti due profili deterministici:

- `large-key-near-max`: lunghezza sostanzialmente fissa vicino al massimo, per
  minimizzare il fanout e massimizzare profondità e propagazione verticale;
- `large-key-half-to-near-max`: lunghezza per-key variabile tra metà del massimo
  e near-max, per introdurre packing e arità irregolari oltre al basso fanout.

Il workload esegue sparse ordered growth, interior fill, più ondate di
delete/regrow plain e poi DUPSORT exact-delete/regrow.  Le chiavi fisiche sono
una funzione deterministica dell'ID logico, così delete e reinsertion ricreano
sempre gli stessi byte.  `mdb_stat()` misura depth, branch pages e leaf pages a
ogni fase e il test richiede una profondità minima.

Con `MDB_HASH_SIZE=32` e soltanto 1536 chiavi primarie nello stato pieno per DB,
il profilo fixed raggiunge depth 5 e quello variabile depth 4.  Il risultato non
è soltanto un albero alto: durante delete/regrow il numero di leaf cambia
fortemente, mostrando che il test attraversa layout e transizioni strutturali
differenti.

Il test ha già aperto due finding distinti:

1. AELMDB completa il profilo fixed, ma nel profilo variabile fallisce durante
   la seconda ondata di delete **plain** (`logical key 1371`) con
   `MDB_CORRUPTED: Located page was wrong type`.  Il failure precede DUPSORT ed
   è quindi distinto da C3.
2. aggmaint completa tutte le ondate plain di entrambi i profili, poi va in
   crash all'ingresso della prima ondata DUPSORT exact-delete.  AddressSanitizer
   localizza l'invalid write nel percorso
   `mdb_node_set_agg_blob -> mdb_node_set_aggval -> mdb_agg_publish_page_up ->
   mdb_agg_dupsort_finish -> _mdb_cursor_del -> mdb_del`.  Questo localizza il
   sottosistema di maintenance interessato, ma non giustifica ancora una causa
   più specifica.

La modalità breve `query/growth/stress` resta invariata e, a H=32, mantiene le
firme precedenti identiche fra AELMDB e aggmaint.  Il nuovo stress è pertanto
mantenuto come target diagnostico separato e non è ancora incluso in
`make check`.  Specifica, telemetry e risultati completi sono in
`docs/compare/large_key_structural_stress.md` e
`results/compare_large_keys.tsv`.  La regressione short H=32 eseguita
con lo stesso harness aggiornato è conservata in
`results/short_regression_06d.tsv`.

