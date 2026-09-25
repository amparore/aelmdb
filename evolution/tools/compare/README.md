# Historical comparison tools

These tools are engineering evidence, not part of the product test interface.
`run_product_vs_first_generation.sh` compares current `src/` with the frozen
first-generation AELMDB reference using the shared deterministic workload.
The normal product test under `tests/compare/` instead checks current output
against frozen expected signatures and does not rebuild historical variants.

`c3_get_both.c` is retained as the focused reproducer for the historical C3
behavior of first-generation AELMDB.
