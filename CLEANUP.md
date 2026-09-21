This review is to help improve the codegenerator as a system and make systemic fixes (no string based hacks/shortcuts)


REMOVE:
- checks that are performance peggiorative: e.g., #pragma omp parallel for schedule(static) reduction(|:unsupported_matrix_format), goes against the prescriptions
- loops with one cycle should not exist in the generated code e.g., for (int d = 0; d < 1; ++d) ..


FIX:
- 1 * 2 should be just 2, multiplications should be evaluated before emission
- default template vector length should be removed, no defaults. `int VS = 16>` shoud be `int VS>`

VERIFY:
- FLOPs model vs IR (see that the loops and ops are properly accounted for and can be correctly derived from the IR), create simple test from user defined materials before verifying on the full code generated suite