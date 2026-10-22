# Laplaciano affine cellwise: campo, geometria e kernel BXYZE

Proposta per un operatore CPU/OpenMP compatibile con il ramo Lagrange di
`SpectralElementLaplacian`: basi tensoriali di grado `p`, esaedri affini anche
obliqui e diffusione scalare `kappa_e` costante in ciascun elemento. L'obiettivo
è sfruttare BXYZE nel calcolo e ridurre il traffico di memoria. Il throughput
va verificato su hardware, gradi e dimensioni di problema definiti.

Per un elemento affine, `x = x0 + J_e * xi` e il Jacobiano è costante. Nel setup
si costruisce la metrica simmetrica

$$
G_e = \kappa_e |\det J_e| J_e^{-1}J_e^{-T}, \qquad
G_e = \begin{bmatrix}
g_{00}&g_{01}&g_{02}\\
g_{01}&g_{11}&g_{12}\\
g_{02}&g_{12}&g_{22}
\end{bmatrix}.
$$

**Bastano sei scalari geometrici/materiali per elemento**: 48 byte in double.
Durante l'apply volumetrico si leggono questi coefficienti già pronti; vertici,
Jacobiano, inversa e determinante servono nel setup. La compressione della
geometria costante è usata anche nella
[struttura matrix-free di deal.II](https://dealii.org/current/doxygen/deal.II/group__matrixfree.html).
Se il materiale varia dentro l'elemento, occorrono inoltre i suoi valori ai
punti di quadratura, oppure il loro calcolo durante l'apply.

Il campo e la geometria usano **gli stessi blocchi e le stesse lane**. Ponendo
`N = p + 1`, `N3 = N*N*N`, `eb = e/B_save` e `b = e%B_save`:

```cpp
v = ix + N * (iy + N * iz);

// Campo e risultato: [blocco][nodo locale][lane].
u[eb * N3 * B_save + v * B_save + b];
r[eb * N3 * B_save + v * B_save + b];

// Geometria compressa: [blocco][componente della metrica][lane].
// c = 0,...,5 identifica g00, g01, g02, g11, g12, g22.
metric[eb * 6 * B_save + c * B_save + b];
```

Sono contigui i valori dello stesso nodo, o della stessa componente geometrica,
appartenenti a elementi diversi. Ogni lane può avere una metrica differente.
La geometria mantiene il batching BXYZE, con sei componenti al posto dei nodi:
non occorre duplicare le coordinate ai punti di quadratura. `Q` indica il numero
di punti di quadratura per direzione.

| Dato | Quantità | Preparazione e durata |
|---|---:|---|
| Campo `u` e risultato `r` | `N3` ciascuno per elemento, più padding | Buffer BXYZE persistenti |
| Metrica `G_e` | 6 per elemento, più padding | Setup; aggiornata se cambiano geometria o materiale |
| Basi `S[q*N+i]`, derivate `D[q*N+i]` | `Q*N` ciascuna | Una copia condivisa per scelta di basi/quadratura |
| Pesi 1D `qw` | `Q` | Condivisi; il peso 3D è il prodotto dei tre pesi |
| Layout e piano DSS | Metadati di blocchi e interfacce | Setup della mesh |
| Workspace | Dipende da `N`, `Q`, `B_exec` | Privato per thread, riutilizzato fra apply |

Il setup separato dall'applicazione segue l'impostazione di partial assembly
descritta da [libCEED](https://libceed.org/en/latest/libCEEDapi/).

Il setup geometrico è disponibile in
[`sfem_dss_bxyze_geometry.hpp`](../operators/dss/sfem_dss_bxyze_geometry.hpp).
Il layout comune `sfem_dss_bxyze_layout_t` si crea da grado, numero di elementi,
`B_save` e `B_exec`; il layout cartesiano esistente ne estende i metadati con
`nx`, `ny`, `nz` e può essere passato direttamente alle nuove funzioni.
Il wrapper mesh alloca un buffer `DSSBXYZEGeometry` con metrica pura (senza
materiale e pesi), per elementi affini con orientamento positivo:

```cpp
sfem::DSSBXYZEGeometry geometry;
const int err = sfem::dss_bxyze_affine_geometry(mesh, layout, &geometry);
// Dopo SFEM_SUCCESS:
// geometry.values->data()[(eb * 6 + c) * layout.element_block_size + b]
```

La routine di setup usa blocchi OpenMP e azzera il padding finale. Il kernel
spectral attuale resta il riferimento: l'uso della metrica precalcolata
nell'apply e le contrazioni su lane sono i passaggi successivi della proposta.

L'applicazione completa procede così:

1. Un thread prende un blocco di elementi e lo attraversa in tile di
   `B_exec` elementi. Legge `u` e `metric` mantenendo `b` contiguo, senza il
   passaggio sistematico BXYZE → buffer scalare di un elemento → BXYZE.
2. La sum factorization con `S` e `D` produce le tre componenti del gradiente
   di riferimento `d` ai punti di quadratura. Si riutilizzano le contrazioni
   intermedie comuni alle tre componenti quando conveniente.
3. Per ciascun punto e ciascuna lane applica `f = G_e * d`, inclusi tutti i
   termini fuori diagonale. La metrica è la stessa per tutti i punti della lane;
   i pesi di quadratura restano separati.
4. Le contrazioni trasposte, con i pesi, riportano i flussi sui nodi e scrivono
   il risultato locale in BXYZE. L'inizializzazione è incorporata nel primo
   contributo, evitando un azzeramento globale dei valori poi sovrascritti.
5. Il DSS somma i contributi dei nodi condivisi e replica la somma nelle copie
   cellwise. Le copie dell'input devono rappresentare un campo continuo.

**Il ciclo sulle lane deve essere vettorizzabile anche dentro le contrazioni**:
i temporanei mantengono `b` come indice più veloce. Vettorizzazione fra celle,
specializzazione per grado e ottimizzazione delle contrazioni 1D sono tecniche
di riferimento per kernel ad alte prestazioni
([Kronbichler e Kormann](https://doi.org/10.1145/3325864)).

`B_save` descrive lo storage; `B_exec` descrive il calcolo e va tarato su SIMD e
cache. Si assegnano preferibilmente blocchi interi ai thread, elaborando i tile
al loro interno per limitare il false sharing. Workspace allineati e riutilizzati,
gestione esplicita delle lane finali e fusione dei passaggi devono contenere
working set e traffico: moltiplicare tutti i temporanei `Q^3` per `B_save` può
essere controproducente. Le lane inattive vanno escluse o inizializzate in modo
coerente, senza influenzare risultato e riduzioni.

Il DSS è distinto dal kernel volumetrico. Il
[DSS attuale](../operators/dss/sfem_dss_bxyze_structured.cpp) richiede topologia
strutturata; per macro-mesh non strutturate serve un piano basato sulla
connettività, comprese le corrispondenze dei nodi sulle interfacce. Se anche il
solver resta in BXYZE, norme e prodotti scalari devono compensare la molteplicità
dei nodi duplicati, e i vincoli devono essere applicati coerentemente.

Per sviluppare e misurare il kernel:

- Conservare la
  [versione spectral cellwise attuale](../operators/dss/sfem_dss_bxyze_spectral_laplacian.cpp)
  come riferimento: oggi usa il microkernel originale su un elemento alla volta
  e ricalcola basi e geometria durante ogni apply.
- Prima mantenere basi, precisione e quadratura `Q=3*p`, spostare i dati statici
  nel setup e introdurre il kernel su lane. Verificare l'equivalenza anche su
  esaedri affini obliqui, dove i termini misti sono attivi.
- Provare separatamente `Q=p+1` Gauss-Legendre: per geometria affine e materiale
  costante integra esattamente i prodotti dei gradienti di basi tensoriali di
  grado `p`, il cui grado per coordinata non supera `2*p`. Una quadratura GLL
  con `p+1` punti ha esattezza differente.
- Misurare volume, DSS e operatore completo; confrontare il riferimento globale
  includendo il suo azzeramento. Esplicitare le conversioni escluse dal tempo
  e l'eventuale precomputazione aggiunta solo a una variante.
- Riportare throughput sui **DOF unici**, memoria occupata e parametri
  `p`, `Q`, `B_save`, `B_exec`, thread e precisione. Usare warmup, ripetizioni,
  problemi di diverse dimensioni e report di vettorizzazione per scegliere
  tile, fusioni e workspace sulla base delle misure.
