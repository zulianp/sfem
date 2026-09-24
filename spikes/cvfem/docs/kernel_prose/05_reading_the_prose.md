## Reading the sections below

The tables above are regenerated from `perf/campaign_grace.csv` on every run. The sections
that follow are not: they are written by hand, they survive a regeneration through
`--prose`, and they quote the measurements that were in front of whoever wrote them.

That matters here because the two no longer share a problem size. The analyses below were
written against 4,121,204 and 8,586,756 dof on an earlier binary; the campaign above sweeps
1,098,500 to 28,756,228 dof and does not contain 4,121,204 at all. So a number in the prose
will often have no counterpart in the tables, and where both exist they were taken on
different binaries months apart.

Nothing below is contradicted by the tables above -- the campaign reproduces the recorded
baseline for the packed residual, 2624 MDOF/s against 2620.2 -- and the reasoning in each
section is about mechanism rather than about a particular figure. But a reader who tries to
find a prose number in a table above will not find it, and that is provenance rather than
disagreement.

One thing the campaign does add that the sections below predate: the layout comparison is
now swept across five sizes, and the assembly ranking INVERTS inside that range. Packed
assembly beats colored at 1,098,500 dof and loses to it by 2x at 28,756,228. Any statement
about which layout wins assembly is a statement about a size.
