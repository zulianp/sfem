# Alps system

srun lscpu
Architecture:                         aarch64
CPU op-mode(s):                       64-bit
Byte Order:                           Little Endian
CPU(s):                               288
On-line CPU(s) list:                  0-287
Vendor ID:                            ARM
Model:                                0
Thread(s) per core:                   1
Core(s) per socket:                   72
Socket(s):                            4
Stepping:                             r0p0
Frequency boost:                      disabled
CPU max MHz:                          3474.0000
CPU min MHz:                          81.0000
BogoMIPS:                             2000.00
Flags:                                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti
L1d cache:                            18 MiB (288 instances)
L1i cache:                            18 MiB (288 instances)
L2 cache:                             288 MiB (288 instances)
L3 cache:                             456 MiB (4 instances)
NUMA node(s):                         36
NUMA node0 CPU(s):                    0-71
NUMA node1 CPU(s):                    72-143
NUMA node2 CPU(s):                    144-215
NUMA node3 CPU(s):                    216-287



https://www.openmp.org/spec-html/5.0/openmpse62.html


Reading OMP_AFFINITY_FORMAT


Short Name	Long Name	Meaning
t	team_num	The value returned by omp_get_team_num().
T	num_teams	The value returned by omp_get_num_teams().
L	nesting_level	The value returned by omp_get_level().
n	thread_num	The value returned by omp_get_thread_num().
N	num_threads	The value returned by omp_get_num_threads().
a	ancestor_tnum	The value returned by omp_get_ancestor_thread_num(level), where level is omp_get_level() minus 1.
H	host	The name for the host machine on which the OpenMP program is running.
P	process_id	The process identifier used by the implementation.
i	native_thread_id	The native thread identifier used by the implementation.
A	thread_affinity	The list of numerical identifiers, in the format of a comma-separated list of integers or integer ranges, that represent processors on which a thread may execute, subject to OpenMP thread affinity control and/or other external affinity mechanisms.


OMP_PROC_BIND=close OMP_NUM_THREADS=72 OMP_PLACES=cores OMP_DISPLAY_ENV=VERBOSE OMP_DISPLAY_AFFINITY=TRUE  srun ./stream_openmp.exe

Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:          481432.0     0.006276     0.005816     0.032947
Scale:         465276.8     0.006551     0.006018     0.033627
Add:           465738.1     0.009520     0.009018     0.009987
Triad:         444682.0     0.009829     0.009445     0.010357

OMP_PROC_BIND=close OMP_NUM_THREADS=144 OMP_PLACES=cores OMP_DISPLAY_ENV=VERBOSE OMP_DISPLAY_AFFINITY=TRUE  srun ./stream_openmp.exe

Your clock granularity/precision appears to be 1 microseconds.
Each test below will take on the order of 7032 microseconds.
   (= 7032 clock ticks)
Increase the size of the arrays if this shows that
you are not getting at least 20 clock ticks per test.
-------------------------------------------------------------
WARNING -- The above is only a rough guideline.
For best results, please be sure you know the
precision of your system timer.
-------------------------------------------------------------
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:          389430.4     0.007337     0.007190     0.008153
Scale:         287710.4     0.010013     0.009732     0.013168
Add:           343674.7     0.012599     0.012221     0.023732
Triad:         340993.7     0.012483     0.012317     0.014424

OMP_PROC_BIND=close OMP_NUM_THREADS=288 OMP_PLACES=cores OMP_DISPLAY_ENV=VERBOSE OMP_DISPLAY_AFFINITY=TRUE  srun ./stream_openmp.exe

Your clock granularity/precision appears to be 1 microseconds.
Each test below will take on the order of 4591 microseconds.
   (= 4591 clock ticks)
Increase the size of the arrays if this shows that
you are not getting at least 20 clock ticks per test.
-------------------------------------------------------------
WARNING -- The above is only a rough guideline.
For best results, please be sure you know the
precision of your system timer.
-------------------------------------------------------------
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:          392383.9     0.007395     0.007136     0.020531
Scale:         488826.3     0.006915     0.005728     0.058679
Add:           598717.9     0.007681     0.007015     0.011307
Triad:         583410.4     0.007882     0.007199     0.031069
-------------------------------------------------------------