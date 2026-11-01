# Future direction: chunked assembly for the quantised BSR store

Status: not started.  Measurements in `RESULTS_QP.md` (section "Upper-only quantised BSR, offsets,
in-place encoding").

## Why

Encoding in place removes the second copy, but the peak of a Newton run with quantised BSR is still
BSR f32's (+10%): HEX8 10.3M dof 7.4 GB against 6.7 GB, while the int8 upper-only store is 0.7 GB
and the element matrices peak at 1.9-2.7 GB.  Two things set the peak, both shared with BSR f32:

1. the f32 assembly on the full pattern (36 B per block, every block of every row);
2. the construction of the node graph (`element_node_graph`), a transient of several GB at 10M.

## Idea

Never hold the whole f32 matrix.  Cut the rows into chunks (about a tenth of the mesh, aligned with
the store's parts and sorting windows) and for each chunk:

- take the elements incident to the chunk's rows (a node-to-element map, or element ranges when the
  numbering is a space-filling curve);
- compute their element matrices directly with the generated kernel (as `ea_store.hpp` does; no
  probing), keeping only the upper blocks (j > i) of rows in the chunk;
- accumulate into a chunk-sized f32 buffer, encode into the records and the remainder, discard.

Peak: store + one chunk (+ graph).  Cost: the elements on chunk borders are computed once per chunk
they touch, about 5% more assembly at 10M with ten chunks; assembly is a small share of a run
(5 s of 60 s for HEX8 int8 at 10M).

## Separately

The node graph's construction transient is a library matter and also lowers BSR f32's peak; it
should be measured and fixed on its own.  The structure the store needs (row lengths, columns of
the upper blocks) could be built per chunk instead of from the full graph.

## Open questions

- Chunk size against recompute and against the parts and windows of the SELL layout.
- Whether the element-to-chunk lists cost more than the recompute they save.
- Device version: chunks map to streams; the encode is already per window.
