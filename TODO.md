# DSS / BXYZE Roadmap

Programma di lavoro per riprodurre, in C/C++ su CPU, le idee principali di
`FlexibleCG.pdf`: storage cell-wise persistente, layout BXYZE, DSS strutturato,
e successiva integrazione con mesh semi-strutturate di SFEM.

## 1. Mesh Strutturata 3D

- [ ] Studiare come SFEM rappresenta mesh strutturate e semi-strutturate.
- [ ] Identificare API e file rilevanti per `sshex8`, `SpectralElementLaplacian` e `SemiStructuredEMLaplacian`.
- [ ] Definire un metadata strutturato minimale per il primo prototipo DSS: `nx`, `ny`, `nz`, `p`, `B`.
- [ ] Definire chiaramente il contratto del modulo DSS: scalar field, HEX tensor-product, 3D, `p` variabile, input/output cell-wise.

## 2. Layout Cell-Wise / BXYZE

- [ ] Implementare una prima conversione da global DoFs a cell-wise element-major.
- [ ] Implementare la conversione da global DoFs a layout BXYZE.
- [ ] Definire macro/funzioni di indexing per BXYZE.
- [ ] Supportare diversi valori di `p`.
- [ ] Supportare diversi valori di `B`.
- [ ] Verificare che copie cell-wise dello stesso DoF globale abbiano valori coerenti.

## 3. Test Di Conversione

- [ ] Aggiungere un test seguendo lo stile della libreria e includendo `sfem_test.hpp`.
- [ ] Testare griglie piccole: `1x1x1`, `2x2x2`, `3x2x2`.
- [ ] Testare `p = 1`, `p = 2`, `p = 3`.
- [ ] Usare dati deterministici, ad esempio `u[i] = i`.
- [ ] Verificare conversione global -> cell-wise element-major.
- [ ] Verificare conversione global -> BXYZE.
- [ ] Aggiungere il test alla build.

## 4. DSS Reference

- [ ] Implementare una DSS reference lenta ma ovviamente corretta: cell-wise -> global accumulate -> scatter back to cell-wise.
- [ ] Usare la DSS reference come oracolo numerico per i test.
- [ ] Verificare la reference con input tutto `1`.
- [ ] Verificare la reference con input random.

## 5. DSS 3D Structured In-Place

- [ ] Implementare DSS structured 3D in-place su layout cell-wise.
- [ ] Implementare passata lungo `x`.
- [ ] Implementare passata lungo `y`.
- [ ] Implementare passata lungo `z`.
- [ ] Garantire ordine sequenziale tra le passate.
- [ ] Supportare `p` variabile.
- [ ] Supportare `B` variabile.
- [ ] Verificare che non ci sia double-counting su facce, spigoli e vertici.
- [ ] Verificare che il risultato fast DSS coincida con DSS reference.

## 6. Test DSS

- [ ] Aggiungere test DSS con vettori random.
- [ ] Confrontare DSS in-place contro DSS reference.
- [ ] Testare `p = 1`, `p = 2`, `p = 3`.
- [ ] Testare diverse dimensioni di griglia.
- [ ] Testare diversi valori di `B`.
- [ ] Testare input tutto `1` e controllare valenze: interno elemento `1`, faccia interna `2`, spigolo interno `4`, vertice interno `8`.
- [ ] Aggiungere test per boundary DoFs con valenza ridotta.

## 7. HPC / Performance

- [ ] Separare benchmark di conversione, DSS only, volumetric only, volumetric + DSS.
- [ ] Misurare throughput in DoF/s cell-wise.
- [ ] Misurare throughput in DoF/s unique.
- [ ] Analizzare bandwidth effettiva.
- [ ] Ottimizzare loop order per BXYZE.
- [ ] Scegliere `B` in funzione di cache e SIMD width.
- [ ] Aggiungere OpenMP sulle interfacce senza race condition.
- [ ] Ridurre false sharing.
- [ ] Valutare in-place vs out-of-place per performance.
- [ ] Rendere il kernel DSS strutturato HPC-ready.

## 8. DSS Optimization

- [ ] Aggiungere fast path per blocchi pieni: evitare divisioni/moduli e branch nel caso comune in cui le run lungo `x` sono allineate a blocchi BXYZE completi.
- [ ] Specializzare il kernel lungo `x`: separare interfacce intra-block (`b -> b+1`) e inter-block (`B-1 -> 0` del blocco successivo).
- [x] Usare una singola regione OpenMP per le tre passate `z`, `y`, `x`, mantenendo le barriere tra assi ma riducendo overhead di creazione/sincronizzazione del team.
- [x] Precomputare offset locali di faccia e `block_stride = B * (p + 1)^3` per ridurre moltiplicazioni nel loop caldo.
- [x] Sostituire gli offset heap-based (`std::vector`) con array fissi su stack per il range `p = 1..7`.
- [ ] Specializzare per ordine polinomiale `p = 1..7`, così `n = p + 1` diventa compile-time e il compilatore può unrollare i loop locali.
- [ ] Valutare specializzazione per `B = 1,2,4,8,16,32,64`, soprattutto per i casi benchmark e per eliminare logica runtime nel path caldo.
- [x] Migliorare loop order e strategia OpenMP: ridurre `collapse(5)` quando produce troppe iterazioni piccole, testando flatten manuale o parallelizzazione su tile/blocchi esterni.
- [ ] Aggiungere misura `SoL` CPU: misurare bandwidth sostenuta con microbenchmark STREAM-like e riportare `sol_useful = gb_s_useful / peak_measured_gb_s`.

## 9. Integrazione SFEM Frontend

- [ ] Collegare i kernel C in `operators/dss` al wrapper C++ `frontend/ops/dss`.
- [ ] Stabilizzare API pubblica del modulo `DSS`.
- [ ] Verificare creazione via `OpFactory` con nome `DSS`.
- [ ] Decidere come esporre conversione global -> cell-wise.
- [ ] Decidere come esporre BXYZE come formato dati persistente.

## 10. Mesh Semi-Strutturate

- [ ] Studiare integrazione con `smesh::to_semistructured`.
- [ ] Studiare indexing locale `sshex8_lidx` e funzioni correlate.
- [ ] Mappare macro-elementi semi-strutturati su storage cell-wise.
- [ ] Implementare conversione global -> BXYZE per mesh semi-strutturate.
- [ ] Portare DSS structured su blocchi semi-strutturati.
- [ ] Aggiungere test su mesh semi-strutturate piccole.
- [ ] Confrontare DSS semi-strutturato contro DSS reference.

## 11. Oltre La Mesh Strutturata

- [ ] Studiare interfacce tra macro-blocchi non strutturati.
- [ ] Disegnare kernel separati per face, edge/line e vertex entities.
- [ ] Gestire orientazioni di facce e inversioni di edge.
- [ ] Evitare double-counting usando una copia rappresentante per macro-blocco.
- [ ] Rimandare hanging nodes/shadow cells a una fase successiva.
