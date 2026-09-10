## Caching the nodal pressure gradient is worth much less than it was

1656 MDOF/s with the gradient hoisted out of the timed loop against 1351 with it rebuilt
inside every apply, on the packed residual — a factor of **1.23**. On the atomic layout,
where the reconstruction is still the flat atomic sweep, the same pair reads 618 against
452, a factor of 1.37.

Both of those used to be far larger: 2.41 and 1.60 on the same two configurations, and
`docs/README_alps.md` records the corresponding solver option (`SFEM_PGRAD_CACHE`) as worth
1.26x off the whole linear solve. Nothing about the caching changed. What changed is the
thing being cached: the reconstruction is 4.9x cheaper than it was, so avoiding it buys
proportionally less.

That is worth stating plainly because it inverts a recommendation. When a pass is slow,
hoisting it out of the inner loop is the obvious lever and it was the right one; once the
pass is fast, the hoist is a modest optimisation carrying a real cost — the cached gradient
has to be invalidated when the state changes, and a stale one is a wrong operator rather
than a slow one. The measurement that justified paying that price no longer says what it
said.

The general lesson is the one this spike keeps re-learning: an optimisation's value is a
property of the code around it, not of the optimisation, and it has to be re-measured after
anything nearby moves.
