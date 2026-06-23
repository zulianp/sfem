The action of the kernels must be tested against a simple harcoded python version (the results must coincide). Use a reference element as an input as well as a deformed one, specialize the test for neohookean and test the operator with zero displacement and shear displacement. Tests include TRI3, TET4, and HEX8 versions of the kernels.


generate_neohookean_ogden_files.py must generate the code for HEX27 as well, the files should be prefixed based on the element type


Refactor the code generation so that SfemKernelDiagnostics has its own file and it is included in the kernel files
also use namespace sfem::codegen as a namespace and rename SfemKernelDiagnostics to KernelDiagnostics


Maintain a minimal high-level code-gen framework usage example of a mooney-rivlin material in python/codegen/framework/docs/ the example includes a python file mooney_rivlin.py, mooney_rivlin.md with the exaplanation, mooney_rivlin.sh to run the code-gen and compile the code to a shared object.





