// The compressed formats on the GPU, behind a plain host interface: the drivers (bench_qp_cuda,
// newton_qp_cuda) see device pointers and this, never the device code (qp_device.cu).
//
// Variants, by the CPU drivers' names:
//   f64, f16, i8md      quadrature-point stores (f64: the uncompressed reference)
//   ead16, eadi8p       element matrices, difference form (eadi8p: int8 with pair scales)
//   bsro16, bsroi8      quantized BSR (f16 with one scale, int8 with one scale per block), from the
//                       device f32 assembly the caller supplies
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qpd {

// The element's connectivity in the kernels' node order and its affine geometry, on the device.
struct Geometry {
    ptrdiff_t nelements = 0;
    const int32_t *elements = nullptr;  // [node][element]
    const float *adj = nullptr;         // [component][element], 9 components
    const float *det = nullptr;         // [element]
    double mu = 1, lmbda = 1;
    // The block-local scatter (attach_blocks): the distinct nodes of each block of 128 consecutive
    // elements, and each element's nodes as indices into its block's list.  Absent: atomicAdd per
    // element node.
    const int32_t *block_ptr = nullptr;    // [block + 1]
    const int32_t *block_nodes = nullptr;  // [block_ptr[block], block_ptr[block + 1])
    const uint16_t *local = nullptr;       // [node][element]
    int max_block_nodes = 0;
};

// Sets g's block-local scatter from the host copy of its connectivity ([node][element]); the
// returned object owns the device arrays.
std::shared_ptr<void> attach_blocks(Geometry &g, const int32_t *h_elements);

// One store.  x, h and out are device vectors with the components interleaved.
class Kernels {
public:
    virtual ~Kernels() = default;
    // The store at state x.
    virtual int update(const double *x) = 0;
    // out += K h.
    virtual int apply(const double *h, double *out) = 0;
    // What the application reads.
    virtual size_t bytes() const = 0;
    // The store's arrays as the host store with 32 lanes holds them, in its order (codes, then
    // scales, then means or pair factors), for comparing the device set-up with the host's.
    virtual int download(std::vector<uint8_t> &codes, std::vector<float> &scales, std::vector<uint8_t> &extra) const {
        (void)codes, (void)scales, (void)extra;
        return 1;
    }
};

const char *element_name();
int element_nodes();

// An element store (f64, f16, i8md, ead16, eadi8p); null for an unknown variant.  On `stream`.
std::unique_ptr<Kernels> create(const std::string &variant, const Geometry &g, void *stream);

// The f32 assembly the quantized BSR encodes: assemble(x, values) writes the BSR values (3x3
// blocks, row-major) of the operator at x on the node graph (rowptr, colidx) the store was made for.
struct Assembly {
    ptrdiff_t n_rows = 0;
    const int32_t *rowptr = nullptr;  // device, sorted rows with the diagonal
    const int32_t *colidx = nullptr;
    const int32_t *h_rowptr = nullptr;  // the same on the host, for the structure
    const int32_t *h_colidx = nullptr;
    std::function<int(const double *x, float *values)> assemble;
};

// A quantized BSR store (bsro16, bsroi8); null for an unknown variant.
std::unique_ptr<Kernels> create_bsr(const std::string &variant, const Assembly &a, void *stream);

}  // namespace qpd
