# Confronto CG globale / cellwise BXYZE

Il driver `frontend/ops/dss/benchmark/sfem_DSSBXYZECGBench.cpp` misura il tempo
alla soluzione dello stesso sistema discreto, senza precondizionatore.

La mesh è una griglia logica di esaedri affini con dimensioni graduate e shear:
gli elementi non sono axis aligned. Si usano gradi 2/4/8, quadratura
Gauss–Legendre con `Q=p+1` e Dirichlet omogenee su tutto il bordo. La soluzione
nodale scelta genera `b=A*truth`; `--rhs smooth` sceglie una funzione liscia,
`--rhs random` valori casuali riproducibili. Il vettore iniziale è zero;
`--initial-scale` permette di impostare una frazione di `truth`.

## Protocollo

- Entrambe le versioni usano `ConjugateGradient`, la stessa ricorrenza, precisione,
  tolleranze (`atol=0`), modalità di fusione e numero di thread.
- I prodotti scalari e gli aggiornamenti fusi passano dalla stessa struttura di
  callback e dai kernel comuni. Solo il cellwise legge i pesi inversi delle
  molteplicità. Le copie nodali e questo lavoro aggiuntivo fanno parte del costo.
- Il globale applica l'operatore elemento per elemento: gather, otto contrazioni
  dirette, metriche/pesi, otto trasposte e scatter. Con più thread lo scatter è
  atomico; l'azzeramento necessario dell'output è incluso nei tempi.
- Il cellwise applica il kernel batched, DSS e condizioni al bordo. `S`, `D`, pesi
  di quadratura e componenti metriche sono gli stessi del globale: le metriche
  sono soltanto riordinate nel layout `[blocco][6][B_save]`.
- Setup, conversioni, reset del vettore iniziale e controlli sono fuori dai tempi.
  Il warmup obbligatorio alloca e riutilizza anche il workspace interno del CG.
  Nella modalità isolata è limitato a `--warmup-iterations` (default 10).
- Si cronometra solo `cg->apply(b,x)`, inclusa l'inizializzazione del residuo.
  Il tempo per iterazione è questo tempo totale diviso per le iterazioni effettive;
  comprende quindi il costo iniziale del solve.
- Ogni campione riparte dallo stesso vettore iniziale. Le ripetizioni sono pari
  e alternano globale/cellwise e cellwise/globale. Si conservano tutti i campioni;
  non si seleziona una coppia B in base ai risultati cronometrati.
- Dopo ogni coppia, un operatore globale seriale valuta il residuo reale per
  entrambe le soluzioni. Il CSV riporta `||b-Ax||/||b-Ax0||`, coerente con la
  normalizzazione del criterio CG. Il driver richiede che sia al massimo `rtol`
  e che le copie cellwise restino identiche. Registra anche l'errore rispetto a
  `truth` e la differenza tra le due soluzioni.

`--deterministic 0` usa riduzioni OpenMP SIMD per i prodotti scalari.
`--deterministic 1` usa le riduzioni BLAS a chunk fissi, comprese le loro
allocazioni temporanee esistenti durante i dot. In entrambi i casi la scelta è
uguale per i due solver. La riduzione del residuo nel kernel fuso resta OpenMP:
il flag non rende deterministico l'intero solve. Ordine delle somme, atomiche e
layout possono produrre numeri di iterazioni leggermente diversi, registrati
separatamente dai tempi per iterazione.

## Esecuzione

Da root del repository, dopo aver compilato il target `sfem_DSSBXYZECGBench`:

```sh
PYTHONPATH=python OMPI_MCA_btl=self build/sfem_DSSBXYZECGBench \
  --levels 2,4,8 --sizes 4 --blocks 8 --exec-blocks 4 \
  --threads 1,8 --fusions 1 --repeat 4 --warmup 1
```

`--help` elenca le opzioni. `--sizes` specifica macroelementi per asse;
`--blocks` e `--exec-blocks` accettano liste. Sono richiesti blocchi e batch pieni.
I valori predefiniti sono piccoli; per riprendere le dimensioni del benchmark
dell'operatore si può impostare `--sizes 16,64`.

Il file `frontend/ops/dss/benchmark/dss_bxyze_cg_bench.csv` contiene i campioni
accoppiati; `dss_bxyze_cg_bench_summary.csv` riporta, per configurazione, minimo
e mediana del tempo totale, mediana del tempo per iterazione, intervallo delle
iterazioni, residui ed errori massimi. Gli speedup sono `globale/cellwise`.
`--output` cambia il percorso di entrambi i file. Un fallimento di convergenza o
validazione termina il driver con errore e non viene registrato come speedup.

## Campagna 16³ / 64³

`run_dss_bxyze_cg_campaign.py` legge le righe `best_apply_dss` del CSV del
benchmark dell'operatore e usa una sola coppia B per grado, dimensione e thread.
Il protocollo predefinito contiene 12 configurazioni: gradi 2/4/8, mesh 16³/64³,
8/1 thread. Esegue quattro ripetizioni su 16³ e due su 64³, alternando l'ordine
globale/cellwise. Ogni metodo e ripetizione viene eseguito in un processo nuovo,
con un warmup di dieci iterazioni, `x0=0`, `rtol=1e-8`, `atol=0`, fusione abilitata
e riduzioni BLAS OpenMP. Il limite comune è un milione di iterazioni. Gli
operatori rimangono quelli descritti sopra, incluso lo scatter atomico globale.

Il driver usa esattamente la graduazione del benchmark dell'operatore:
`a=x+0.125*x*x`, `b=0.75*y+0.0625*y*y`, `c=0.5*z+0.03125*z*z`, prima dello shear.
Gli array geometrici nodali vengono liberati dopo il setup; nel processo
cellwise la connettività viene liberata prima del solve e ricreata per la
validazione, dopo aver liberato i vettori interni del CG. La memoria massima
durante il solve e quella dell'intero processo sono registrate separatamente.

Il Python scelto per questa sessione è `/private/tmp/sfem_gradient_plot_venv/bin/python`;
il percorso storico indicato in AGENTS.md non esiste su questa macchina.

```sh
PYTHONPATH=python /private/tmp/sfem_gradient_plot_venv/bin/python -u \
  frontend/ops/dss/benchmark/run_dss_bxyze_cg_campaign.py --detach
```

Il processo continua dopo la chiusura del comando di lancio. Lo script salva:

- `dss_bxyze_cg_16_64_bench.csv`: ogni singolo solve, con stato, tempo, iterazioni,
  residuo reale, errore rispetto a `truth`, coerenza delle copie e memoria.
- `dss_bxyze_cg_16_64_bench_summary.csv`: tutte le 12 configurazioni, stato di
  avanzamento, mediane e speedup, pubblicati solo a campionamento completo.
- `dss_bxyze_cg_16_64_bench.log` e `..._status.json`: avanzamento e processo attivo.
- `dss_bxyze_cg_16_64_bench_runs/`: log e CSV per ciascun processo, protocollo e
  copia dell'eseguibile usato, con SHA-256, per proteggerlo da successive build.

Rieseguendo lo stesso comando, lo script riprende i campioni già registrati nel
CSV senza ripeterli; un lock impedisce due campagne contemporanee sullo stesso
output. Per una nuova campagna usare un altro `--output-base`. I solve non
convergenti o con validazione fallita rimangono nel CSV con stato esplicito e
senza un tempo alla soluzione valido.

## Campagna a iterazioni fisse per ordine 8

L'opzione `--fixed-iterations K` è disponibile nella modalità isolata e imposta
`max_it=K`, `rtol=0`. Il run è valido quando esegue esattamente `K` iterazioni,
anche se il residuo non ha raggiunto una tolleranza di convergenza. Il CSV
registra `status=fixed_iterations`, il tempo in `attempt_seconds`, il tempo per
iterazione e il residuo vero ricalcolato fuori dal timer. `tts_s` resta `NaN`:
questo esperimento misura il costo del CG a `K` passi, non il tempo alla soluzione.
Si controllano comunque finitezza, Dirichlet e uguaglianza delle copie cellwise.

La campagna usa ordine 8, quattro ripetizioni per metodo su 16³ con `K=5000` e
due ripetizioni su 64³ con `K=200`; entrambe le versioni usano lo stesso `K`
all'interno di ogni configurazione. Il runner salva la mediana del tempo e del
residuo vero, oltre al rapporto dei residui cellwise/globale. Gli output hanno
nomi distinti dalla campagna a convergenza:
`dss_bxyze_cg_p8_fixed5000_n16*` e `dss_bxyze_cg_p8_fixed200_n64*`.

```sh
PYTHONPATH=python /private/tmp/sfem_gradient_plot_venv/bin/python \
  frontend/ops/dss/benchmark/run_dss_bxyze_cg_campaign.py \
  --levels 8 --sizes 16 --threads 8,1 --fixed-iterations 5000 \
  --output-base frontend/ops/dss/benchmark/dss_bxyze_cg_p8_fixed5000_n16 --detach
```

Il secondo run usa `--sizes 64 --fixed-iterations 200` e l'output base
`dss_bxyze_cg_p8_fixed200_n64`. Per evitare interferenza nei tempi, va eseguito
dopo il completamento del primo.

## Grafici

```sh
PYTHONPATH=python /private/tmp/sfem_gradient_plot_venv/bin/python -u \
  frontend/ops/dss/benchmark/plot_dss_bxyze_cg.py --between-solves
```

Lo script genera, per ciascuna mesh e numero di thread, solo il PNG del throughput
per iterazione `N_global / median(T_CG / iterations)`. Usa gli stessi colori, griglia e stile
degli altri plot; sopra la barra cellwise indica lo speedup nella forma `1.23 X`.
Sono disponibili anche filtri `--size` e `--threads`.
`--completed-only` include solo le configurazioni completate, rimuovendo anche
le colonne vuote degli ordini non eseguiti; è adatto ai risultati di una campagna
interrotta.

Vengono rappresentate solo configurazioni con tutte le ripetizioni completate
e validate. Gli ordini ancora da completare sono indicati come `pending`, senza
barre né speedup. Ogni esecuzione legge un'unica versione del riepilogo e salva
lo snapshot usato in `dss_bxyze_cg_plot_data.csv`.

Durante una campagna attiva, `--between-solves` sospende soltanto il coordinatore,
lascia terminare il solve già avviato, genera le figure e riattiva il coordinatore
anche in caso di errore. Il processo cronometrato non viene sospeso e Matplotlib
viene importato solo dopo il suo completamento. Il limite d'attesa predefinito è
900 secondi, modificabile con `--wait-seconds`. Senza questa opzione lo script
genera subito i grafici; usarlo quando non sono in corso misure.
