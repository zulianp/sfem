from codegen.framework.plans.conventions import restrict_prelude
from codegen.framework.targets import current_target


def _matching_brace_index(text, open_brace):
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise RuntimeError("unmatched brace while expanding tensor-product kernels")


def _residual_stream_method_end(text, start):
    open_brace = text.find("{", start)
    if open_brace < 0:
        raise RuntimeError("missing residual stream method body")
    close_brace = _matching_brace_index(text, open_brace)
    semicolon = close_brace + 1
    while semicolon < len(text) and text[semicolon].isspace():
        if text[semicolon] == "\n":
            break
        semicolon += 1
    return close_brace + 1


def _expand_residual_stream_method(method):
    pointer_method = method.replace(
        "template <int NC, typename StreamContainer>",
        "template <int NC>",
        1,
    )
    pointer_method = pointer_method.replace(
        "const StreamContainer streams,",
        "const s_t *const RSTR streams[NC * NS],",
        1,
    )
    pointer_method = pointer_method.replace(
        "StreamContainer output)",
        "s_t *const RSTR output[NC * NS])",
        1,
    )

    contiguous_method = method.replace(
        "template <int NC, typename StreamContainer>",
        "template <int NC>",
        1,
    )
    contiguous_method = contiguous_method.replace(
        "void evaluate(",
        "void evaluate_contiguous(",
        1,
    )
    contiguous_method = contiguous_method.replace(
        "void evaluate_value(",
        "void evaluate_value_contiguous(",
        1,
    )
    contiguous_method = contiguous_method.replace(
        "void integrate(",
        "void integrate_contiguous(",
        1,
    )
    contiguous_method = contiguous_method.replace(
        "void integrate_value(",
        "void integrate_value_contiguous(",
        1,
    )
    contiguous_method = contiguous_method.replace(
        "const StreamContainer streams,",
        "const s_t streams[NC * NS][VS],",
        1,
    )
    contiguous_method = contiguous_method.replace(
        "StreamContainer output)",
        "s_t output[NC * NS][VS])",
        1,
    )
    return "%s\n\n%s" % (pointer_method, contiguous_method)


def _expand_residual_stream_layouts(header):
    generic_template = "  template <int NC, typename StreamContainer>\n"
    chunks = []
    cursor = 0
    while True:
        start = header.find(generic_template, cursor)
        if start < 0:
            chunks.append(header[cursor:])
            return "".join(chunks)
        end = _residual_stream_method_end(header, start)
        chunks.append(header[cursor:start])
        chunks.append(_expand_residual_stream_method(header[start:end]))
        cursor = end


def _work_item_loop_text(indent, index_name, simd_lines, single_work_item):
    if single_work_item:
        return "%s{" % indent
    return "%s\n%sfor (int %s = 0; %s < ne; ++%s) {" % (
        "\n".join("%s%s" % (indent, line) for line in simd_lines),
        indent,
        index_name,
        index_name,
        index_name,
    )


def _weak_form_ops_source(
    inline_qualifier,
    work_item_index,
    simd_lines,
    single_work_item,
    *,
    lane_blocked,
):
    """The weak-form micro-kernels, spelled for a block of work items or for one.

    The sum factorization is written once.  A kernel that walks a block of
    elements at a time wants the lane-blocked spelling: `ne` work items, stage
    buffers with a slot per lane, and a lane loop the target may vectorize.  A
    kernel that has one element in hand -- matrix assembly, which scatters into
    a sparse row and has nowhere to put a block -- wants the same arithmetic
    with none of that: no `ne`, no lane in the stage index, no loop.

    Both are this one body.  Rendering it twice is what keeps the second from
    being a transcription of the first that drifts.
    """
    values = {
        "inline_qualifier": inline_qualifier,
        "work_item": work_item_index,
        "weak_ops": "TensorProductWeakOps" if lane_blocked else "TensorProductWeakOpsScalar",
        "scalar_suffix": "" if lane_blocked else "_scalar",
        "lane_stride": " * VS" if lane_blocked else "",
        "lane_offset": (" * VS + %s" % work_item_index) if lane_blocked else "",
        "work_item_count_6": "      const int ne,\n" if lane_blocked else "",
        "work_item_count_4": "    const int ne,\n" if lane_blocked else "",
        "ne_argument": "ne, " if lane_blocked else "",
    }
    for indent_size in (12, 16, 20):
        values["work_item_loop_%d" % indent_size] = _work_item_loop_text(
            " " * indent_size,
            work_item_index,
            simd_lines,
            single_work_item or not lane_blocked,
        )
    return _WEAK_FORM_OPS_TEMPLATE % values


def _restrict_define_line(restrict_definition):
    restrict_definition = str(restrict_definition)
    if restrict_definition:
        return "#define RSTR %s" % restrict_definition
    return "#define RSTR"


def tensor_product_kernels_header_source_for(target):
    """`tensor_product_kernels`, spelled for one target.

    The twin of `geometry_kernels_header_source_for`, and it exists for the
    same reason: the residual emitter called the builder with its CPU defaults,
    and a device target's work item is the literal `0`, so what came out was
    `for (int 0 = 0; 0 < ne; ++0)`.  One place turns a target into this header.
    """
    inline_qualifier = target.inline_qualifier()
    vectorize_pragma = target.vectorize_pragma()
    policy = target.loop_lowering_policy()
    return sfem_tensor_product_kernels_header_source(
        inline_qualifier=inline_qualifier,
        define_sfem_inline=inline_qualifier == "SFEM_INLINE",
        restrict_definition=target.restrict_definition(),
        work_item_index=target.work_item_index(),
        simd_lines=() if vectorize_pragma is None else (vectorize_pragma,),
        single_work_item=not policy.emits_lane_loop,
        header_guard_suffix=target.header_guard_suffix(),
    )


def sfem_tensor_product_kernels_header_source(
    *,
    inline_qualifier=None,
    inline_definition="inline",
    define_sfem_inline=True,
    restrict_definition="",
    work_item_index=None,
    simd_lines=None,
    single_work_item=False,
    header_guard_suffix="HPP",
):
    target = current_target()
    inline_qualifier = target.inline_qualifier() if inline_qualifier is None else inline_qualifier
    work_item_index = target.work_item_index() if work_item_index is None else work_item_index
    if simd_lines is None:
        pragma = target.vectorize_pragma()
        simd_lines = () if pragma is None else (pragma,)
    values = {
        # Empty on a CPU target, so these keep the spelling they have always
        # had; `__host__ __device__` on a device target, because a kernel calls
        # them and nvcc otherwise wants --expt-relaxed-constexpr to look away.
        "kernel_callable": (
            "%s " % target.kernel_callable_qualifier()
            if target.kernel_callable_qualifier()
            else ""
        ),
        "header_guard_suffix": header_guard_suffix,
        "sfem_inline_block": (
            "%s\n\n" % "\n".join(target.inline_definition_lines(inline_definition))
            if define_sfem_inline
            else ""
        ),
        "inline_qualifier": inline_qualifier,
        "inline_definition": inline_definition,
        "restrict_prelude": "\n".join(restrict_prelude(restrict_definition or "")),
        "work_item": work_item_index,
    }
    for indent_size in (12, 16, 20):
        indent = " " * indent_size
        values["work_item_loop_%d" % indent_size] = _work_item_loop_text(
            indent,
            work_item_index,
            simd_lines,
            single_work_item,
        )
    values["weak_form_ops"] = "\n\n".join(
        _expand_residual_stream_layouts(
            _weak_form_ops_source(
                inline_qualifier,
                work_item_index if lane_blocked else "0",
                simd_lines,
                single_work_item,
                lane_blocked=lane_blocked,
            )
        )
        for lane_blocked in (True, False)
    )
    return _expand_residual_stream_layouts(_TENSOR_PRODUCT_KERNELS_TEMPLATE % values)


# The reductions in `test` and `integrate` keep their index arithmetic, and it
# was measured rather than argued.  They accumulate into registers across the
# quadrature direction and store once, so the base of each read moves with the
# reduction variable and cannot be named outside the lane loop.  Naming it would
# mean interchanging the lane and reduction loops, which turns one register
# accumulation into a read-modify-write through the stage buffer.
#
# Interchanging one of the nine stages -- the first of the three-dimensional
# `test` -- cost, on one Grace node at 204.8 MDOF with a 20 GB working set and
# OMP_PROC_BIND=true: 7.7% to 14.4% single threaded across three interleaved
# pairs, 4.8% at 32 threads, and 0.6% at 72 where the kernel is memory bound.
# Answers were identical throughout, so the whole of it was cost.
#
# Everything whose base does *not* move with the reduction -- every store, and
# the reads the surrounding loops already fix -- is named outside the loop.

_WEAK_FORM_OPS_TEMPLATE = r'''template <typename s_t, int NQ, int NS, int VS, int ND>
struct %(weak_ops)s;

template <typename s_t, int NQ, int NS, int VS>
struct %(weak_ops)s<s_t, NQ, NS, VS, 2> {
  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void gradient_impl(
%(work_item_count_6)s      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const StreamContainer streams,
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t value_x[NQ1 * NS1%(lane_stride)s];
    s_t grad_x[NQ1 * NS1%(lane_stride)s];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
%(work_item_loop_16)s
          s_t v = s_t(0);
          s_t gx = s_t(0);
          for (int sx = 0; sx < NS1; ++sx) {
            const int shape = sx + NS1 * sy;
            const s_t u = streams[shape * NC + component][%(work_item)s];
            v += u * shape_1d[qx * NS1 + sx];
            gx += u * grad_1d[qx * NS1 + sx];
          }
          const int i = (qx * NS1 + sy)%(lane_offset)s;
          value_x[i] = v;
          grad_x[i] = gx;
        }
      }
    }
    for (int qy = 0; qy < NQ1; ++qy) {
      for (int qx = 0; qx < NQ1; ++qx) {
        const int q = qx + NQ1 * qy;
        s_t *const RSTR gradient_q0 = &gradient[(q * 2 + 0)%(lane_stride)s];
        s_t *const RSTR gradient_q1 = &gradient[(q * 2 + 1)%(lane_stride)s];
%(work_item_loop_16)s
          s_t gx = s_t(0);
          s_t gy = s_t(0);
          for (int sy = 0; sy < NS1; ++sy) {
            const int i = (qx * NS1 + sy)%(lane_offset)s;
            gx += grad_x[i] * shape_1d[qy * NS1 + sy];
            gy += value_x[i] * grad_1d[qy * NS1 + sy];
          }
          gradient_q0[%(work_item)s] = gx;
          gradient_q1[%(work_item)s] = gy;
        }
      }
    }
  }

  template <int NC>
  static %(inline_qualifier)s void gradient(
%(work_item_count_6)s      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NS * NC],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(%(ne_argument)sshape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static %(inline_qualifier)s void gradient_contiguous(
%(work_item_count_6)s      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NS * NC][VS],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(%(ne_argument)sshape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static %(inline_qualifier)s void test(
%(work_item_count_6)s      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR flux,
      s_t *const RSTR out_streams[NS * NC],
      const int component) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t stage_x[NQ1 * NS1%(lane_stride)s];
    s_t stage_y[NQ1 * NS1%(lane_stride)s];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
%(work_item_loop_16)s
          s_t tx = s_t(0);
          s_t ty = s_t(0);
          for (int qy = 0; qy < NQ1; ++qy) {
            const int q = qx + NQ1 * qy;
            tx += flux[(q * 2 + 0)%(lane_offset)s] * shape_1d[qy * NS1 + sy];
            ty += flux[(q * 2 + 1)%(lane_offset)s] * grad_1d[qy * NS1 + sy];
          }
          const int i = (qx * NS1 + sy)%(lane_offset)s;
          stage_x[i] = tx;
          stage_y[i] = ty;
        }
      }
    }
    for (int sy = 0; sy < NS1; ++sy) {
      for (int sx = 0; sx < NS1; ++sx) {
        const int shape = sx + NS1 * sy;
%(work_item_loop_16)s
          s_t value = s_t(0);
          for (int qx = 0; qx < NQ1; ++qx) {
            const int i = (qx * NS1 + sy)%(lane_offset)s;
            value += stage_x[i] * grad_1d[qx * NS1 + sx]
                               + stage_y[i] * shape_1d[qx * NS1 + sx];
          }
          out_streams[shape * NC + component][%(work_item)s] += value;
        }
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int VS>
struct %(weak_ops)s<s_t, NQ, NS, VS, 3> {
  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void gradient_impl(
%(work_item_count_6)s      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const StreamContainer streams,
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t value_x[NQ1 * NS1 * NS1%(lane_stride)s];
    s_t grad_x[NQ1 * NS1 * NS1%(lane_stride)s];
    s_t value_xy[NQ1 * NQ1 * NS1%(lane_stride)s];
    s_t grad_x_xy[NQ1 * NQ1 * NS1%(lane_stride)s];
    s_t grad_y_xy[NQ1 * NQ1 * NS1%(lane_stride)s];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sz = 0; sz < NS1; ++sz) {
%(work_item_loop_20)s
            s_t v = s_t(0);
            s_t gx = s_t(0);
            for (int sx = 0; sx < NS1; ++sx) {
              const int shape = sx + NS1 * (sy + NS1 * sz);
              const s_t u = streams[shape * NC + component][%(work_item)s];
              v += u * shape_1d[qx * NS1 + sx];
              gx += u * grad_1d[qx * NS1 + sx];
            }
            const int i = ((qx * NS1 + sy) * NS1 + sz)%(lane_offset)s;
            value_x[i] = v;
            grad_x[i] = gx;
          }
        }
      }
    }
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int sz = 0; sz < NS1; ++sz) {
%(work_item_loop_20)s
            s_t v = s_t(0);
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            for (int sy = 0; sy < NS1; ++sy) {
              const int i = ((qx * NS1 + sy) * NS1 + sz)%(lane_offset)s;
              v += value_x[i] * shape_1d[qy * NS1 + sy];
              gx += grad_x[i] * shape_1d[qy * NS1 + sy];
              gy += value_x[i] * grad_1d[qy * NS1 + sy];
            }
            const int j = ((qx * NQ1 + qy) * NS1 + sz)%(lane_offset)s;
            value_xy[j] = v;
            grad_x_xy[j] = gx;
            grad_y_xy[j] = gy;
          }
        }
      }
    }
    for (int qz = 0; qz < NQ1; ++qz) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int qx = 0; qx < NQ1; ++qx) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          s_t *const RSTR gradient_q0 = &gradient[(q * 3 + 0)%(lane_stride)s];
          s_t *const RSTR gradient_q1 = &gradient[(q * 3 + 1)%(lane_stride)s];
          s_t *const RSTR gradient_q2 = &gradient[(q * 3 + 2)%(lane_stride)s];
%(work_item_loop_20)s
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            s_t gz = s_t(0);
            for (int sz = 0; sz < NS1; ++sz) {
              const int j = ((qx * NQ1 + qy) * NS1 + sz)%(lane_offset)s;
              gx += grad_x_xy[j] * shape_1d[qz * NS1 + sz];
              gy += grad_y_xy[j] * shape_1d[qz * NS1 + sz];
              gz += value_xy[j] * grad_1d[qz * NS1 + sz];
            }
            gradient_q0[%(work_item)s] = gx;
            gradient_q1[%(work_item)s] = gy;
            gradient_q2[%(work_item)s] = gz;
          }
        }
      }
    }
  }

  template <int NC>
  static %(inline_qualifier)s void gradient(
%(work_item_count_6)s      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NS * NC],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(%(ne_argument)sshape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static %(inline_qualifier)s void gradient_contiguous(
%(work_item_count_6)s      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NS * NC][VS],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(%(ne_argument)sshape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static %(inline_qualifier)s void test(
%(work_item_count_6)s      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR flux,
      s_t *const RSTR out_streams[NS * NC],
      const int component) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t stage_x[NQ1 * NQ1 * NS1%(lane_stride)s];
    s_t stage_y[NQ1 * NQ1 * NS1%(lane_stride)s];
    s_t stage_z[NQ1 * NQ1 * NS1%(lane_stride)s];
    s_t stage_xy_x[NQ1 * NS1 * NS1%(lane_stride)s];
    s_t stage_xy_y[NQ1 * NS1 * NS1%(lane_stride)s];
    s_t stage_xy_z[NQ1 * NS1 * NS1%(lane_stride)s];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int sz = 0; sz < NS1; ++sz) {
%(work_item_loop_20)s
            s_t tx = s_t(0);
            s_t ty = s_t(0);
            s_t tz = s_t(0);
            for (int qz = 0; qz < NQ1; ++qz) {
              const int q = qx + NQ1 * (qy + NQ1 * qz);
              tx += flux[(q * 3 + 0)%(lane_offset)s] * shape_1d[qz * NS1 + sz];
              ty += flux[(q * 3 + 1)%(lane_offset)s] * shape_1d[qz * NS1 + sz];
              tz += flux[(q * 3 + 2)%(lane_offset)s] * grad_1d[qz * NS1 + sz];
            }
            const int i = ((qx * NQ1 + qy) * NS1 + sz)%(lane_offset)s;
            stage_x[i] = tx;
            stage_y[i] = ty;
            stage_z[i] = tz;
          }
        }
      }
    }
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sz = 0; sz < NS1; ++sz) {
%(work_item_loop_20)s
            s_t tx = s_t(0);
            s_t ty = s_t(0);
            s_t tz = s_t(0);
            for (int qy = 0; qy < NQ1; ++qy) {
              const int i = ((qx * NQ1 + qy) * NS1 + sz)%(lane_offset)s;
              tx += stage_x[i] * shape_1d[qy * NS1 + sy];
              ty += stage_y[i] * grad_1d[qy * NS1 + sy];
              tz += stage_z[i] * shape_1d[qy * NS1 + sy];
            }
            const int j = ((qx * NS1 + sy) * NS1 + sz)%(lane_offset)s;
            stage_xy_x[j] = tx;
            stage_xy_y[j] = ty;
            stage_xy_z[j] = tz;
          }
        }
      }
    }
    for (int sz = 0; sz < NS1; ++sz) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sx = 0; sx < NS1; ++sx) {
          const int shape = sx + NS1 * (sy + NS1 * sz);
%(work_item_loop_20)s
            s_t value = s_t(0);
            for (int qx = 0; qx < NQ1; ++qx) {
              const int j = ((qx * NS1 + sy) * NS1 + sz)%(lane_offset)s;
              value += stage_xy_x[j] * grad_1d[qx * NS1 + sx]
                                   + (stage_xy_y[j] + stage_xy_z[j]) * shape_1d[qx * NS1 + sx];
            }
            out_streams[shape * NC + component][%(work_item)s] += value;
          }
        }
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int VS, int ND, int NC = ND>
static %(inline_qualifier)s void tensor_gradient%(scalar_suffix)s(
%(work_item_count_4)s    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR streams[NS * NC],
    const int component,
    s_t *const RSTR gradient) {
  %(weak_ops)s<s_t, NQ, NS, VS, ND>::template gradient<NC>(
      %(ne_argument)sshape_1d, grad_1d, streams, component, gradient);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC = ND>
static %(inline_qualifier)s void tensor_gradient_contiguous%(scalar_suffix)s(
%(work_item_count_4)s    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t streams[NS * NC][VS],
    const int component,
    s_t *const RSTR gradient) {
  %(weak_ops)s<s_t, NQ, NS, VS, ND>::template gradient_contiguous<NC>(
      %(ne_argument)sshape_1d, grad_1d, streams, component, gradient);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC = ND>
static %(inline_qualifier)s void tensor_test%(scalar_suffix)s(
%(work_item_count_4)s    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR flux,
    s_t *const RSTR out_streams[NS * NC],
    const int component) {
  %(weak_ops)s<s_t, NQ, NS, VS, ND>::template test<NC>(
      %(ne_argument)sshape_1d, grad_1d, flux, out_streams, component);
}'''


_TENSOR_PRODUCT_KERNELS_TEMPLATE = r'''#ifndef SFEM_CODEGEN_TENSOR_PRODUCT_KERNELS_%(header_guard_suffix)s
#define SFEM_CODEGEN_TENSOR_PRODUCT_KERNELS_%(header_guard_suffix)s

#include <stddef.h>

%(sfem_inline_block)s
%(restrict_prelude)s

namespace sfem {
namespace codegen {

static %(kernel_callable)sconstexpr int ipow(const int base, const int exponent) {
  return exponent == 0 ? 1 : base * ipow(base, exponent - 1);
}

static %(kernel_callable)sconstexpr int integer_root_search(const int value, const int exponent, const int candidate) {
  return ipow(candidate, exponent) >= value ? candidate : integer_root_search(value, exponent, candidate + 1);
}

static %(kernel_callable)sconstexpr int integer_root(const int value, const int exponent) {
  return integer_root_search(value, exponent, 1);
}

%(weak_form_ops)s

template <typename s_t, int NQ, int NS, int VS, int ND>
struct TensorProductResidualOps;

template <typename s_t, int NQ, int NS, int VS>
struct TensorProductResidualOps<s_t, NQ, NS, VS, 2> {
  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void evaluate(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const StreamContainer streams,
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1 * VS];
    s_t gx[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
%(work_item_loop_12)s
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          const s_t u = streams[s * NC + f][%(work_item)s];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + %(work_item)s;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 2 + 0) * VS];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 2 + 1) * VS];
%(work_item_loop_12)s
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + %(work_item)s;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        value_q[%(work_item)s] = v;
        gradient_q0[%(work_item)s] = g0;
        gradient_q1[%(work_item)s] = g1;
      }
    }
  }

  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void evaluate_value(
      const int ne,
      const s_t *const shape_1d,
      const StreamContainer streams,
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR vx_q = &vx[((f * NQ1 + qx) * NS1 + sy) * VS];
%(work_item_loop_12)s
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          v += streams[s * NC + f][%(work_item)s] * shape_1d[qx * NS1 + sx];
        }
        vx_q[%(work_item)s] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
%(work_item_loop_12)s
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[((f * NQ1 + qx) * NS1 + sy) * VS + %(work_item)s] * shape_1d[qy * NS1 + sy];
        }
        value_q[%(work_item)s] = v;
      }
    }
  }

  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void integrate(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      StreamContainer output) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1 * VS];
    s_t sg[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
%(work_item_loop_12)s
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q) * VS + %(work_item)s] * shape_1d[qy * NS1 + sy]
                       + grad_coeff[((f * NQ + q) * 2 + 1) * VS + %(work_item)s] * grad_1d[qy * NS1 + sy];
          b += grad_coeff[((f * NQ + q) * 2 + 0) * VS + %(work_item)s] * shape_1d[qy * NS1 + sy];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + %(work_item)s;
        sv[i] = a;
        sg[i] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
%(work_item_loop_12)s
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + %(work_item)s;
          v += sv[i] * shape_1d[qx * NS1 + sx] + sg[i] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f][%(work_item)s] += v;
      }
    }
  }

  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void integrate_value(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      StreamContainer output) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR sv_q = &sv[((f * NQ1 + qx) * NS1 + sy) * VS];
%(work_item_loop_12)s
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q) * VS + %(work_item)s] * shape_1d[qy * NS1 + sy];
        }
        sv_q[%(work_item)s] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
%(work_item_loop_12)s
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += sv[((f * NQ1 + qx) * NS1 + sy) * VS + %(work_item)s] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f][%(work_item)s] += v;
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int VS>
struct TensorProductResidualOps<s_t, NQ, NS, VS, 3> {
  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void evaluate(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const StreamContainer streams,
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1 * VS];
    s_t gx[NC * NQ1 * NS1 * NS1 * VS];
    s_t vxy[NC * NQ1 * NQ1 * NS1 * VS];
    s_t g0xy[NC * NQ1 * NQ1 * NS1 * VS];
    s_t g1xy[NC * NQ1 * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
%(work_item_loop_12)s
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          const s_t u = streams[s * NC + f][%(work_item)s];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + %(work_item)s;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
%(work_item_loop_12)s
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + %(work_item)s;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + %(work_item)s;
        vxy[j] = v;
        g0xy[j] = g0;
        g1xy[j] = g1;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 3 + 0) * VS];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 3 + 1) * VS];
      s_t *const RSTR gradient_q2 = &gradient[((f * NQ + q) * 3 + 2) * VS];
%(work_item_loop_12)s
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        s_t g2 = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + %(work_item)s;
          v += vxy[j] * shape_1d[qz * NS1 + sz];
          g0 += g0xy[j] * shape_1d[qz * NS1 + sz];
          g1 += g1xy[j] * shape_1d[qz * NS1 + sz];
          g2 += vxy[j] * grad_1d[qz * NS1 + sz];
        }
        value_q[%(work_item)s] = v;
        gradient_q0[%(work_item)s] = g0;
        gradient_q1[%(work_item)s] = g1;
        gradient_q2[%(work_item)s] = g2;
      }
    }
  }

  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void evaluate_value(
      const int ne,
      const s_t *const shape_1d,
      const StreamContainer streams,
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1 * VS];
    s_t vxy[NC * NQ1 * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vx_q = &vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS];
%(work_item_loop_12)s
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          v += streams[s * NC + f][%(work_item)s] * shape_1d[qx * NS1 + sx];
        }
        vx_q[%(work_item)s] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vxy_q = &vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS];
%(work_item_loop_12)s
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + %(work_item)s] * shape_1d[qy * NS1 + sy];
        }
        vxy_q[%(work_item)s] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
%(work_item_loop_12)s
        s_t v = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          v += vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + %(work_item)s] * shape_1d[qz * NS1 + sz];
        }
        value_q[%(work_item)s] = v;
      }
    }
  }

  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void integrate(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      StreamContainer output) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1 * VS];
    s_t z1[NC * NQ1 * NQ1 * NS1 * VS];
    s_t z2[NC * NQ1 * NQ1 * NS1 * VS];
    s_t yz0[NC * NQ1 * NS1 * NS1 * VS];
    s_t yz1[NC * NQ1 * NS1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
%(work_item_loop_12)s
        s_t a = s_t(0);
        s_t b = s_t(0);
        s_t c = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q) * VS + %(work_item)s] * shape_1d[qz * NS1 + sz]
                       + grad_coeff[((f * NQ + q) * 3 + 2) * VS + %(work_item)s] * grad_1d[qz * NS1 + sz];
          b += grad_coeff[((f * NQ + q) * 3 + 0) * VS + %(work_item)s] * shape_1d[qz * NS1 + sz];
          c += grad_coeff[((f * NQ + q) * 3 + 1) * VS + %(work_item)s] * shape_1d[qz * NS1 + sz];
        }
        const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + %(work_item)s;
        z0[i] = a;
        z1[i] = b;
        z2[i] = c;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
%(work_item_loop_12)s
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + %(work_item)s;
          a += z0[i] * shape_1d[qy * NS1 + sy] + z2[i] * grad_1d[qy * NS1 + sy];
          b += z1[i] * shape_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + %(work_item)s;
        yz0[j] = a;
        yz1[j] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
%(work_item_loop_12)s
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + %(work_item)s;
          v += yz0[j] * shape_1d[qx * NS1 + sx] + yz1[j] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f][%(work_item)s] += v;
      }
    }
  }

  template <int NC, typename StreamContainer>
  static %(inline_qualifier)s void integrate_value(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      StreamContainer output) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1 * VS];
    s_t yz0[NC * NQ1 * NS1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR z0_q = &z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS];
%(work_item_loop_12)s
        s_t a = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q) * VS + %(work_item)s] * shape_1d[qz * NS1 + sz];
        }
        z0_q[%(work_item)s] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR yz0_q = &yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS];
%(work_item_loop_12)s
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          a += z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + %(work_item)s] * shape_1d[qy * NS1 + sy];
        }
        yz0_q[%(work_item)s] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
%(work_item_loop_12)s
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + %(work_item)s] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f][%(work_item)s] += v;
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static %(inline_qualifier)s void tensor_evaluate(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t *const RSTR streams[NC * NS],
    s_t *const value,
    s_t *const gradient) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template evaluate<NC>(
      ne, shape_1d, grad_1d, streams, value, gradient);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static %(inline_qualifier)s void tensor_evaluate_contiguous(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t streams[NC * NS][VS],
    s_t *const value,
    s_t *const gradient) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template evaluate_contiguous<NC>(
      ne, shape_1d, grad_1d, streams, value, gradient);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static %(inline_qualifier)s void tensor_evaluate_value(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const RSTR streams[NC * NS],
    s_t *const value) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template evaluate_value<NC>(
      ne, shape_1d, streams, value);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static %(inline_qualifier)s void tensor_evaluate_value_contiguous(
    const int ne,
    const s_t *const shape_1d,
    const s_t streams[NC * NS][VS],
    s_t *const value) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template evaluate_value_contiguous<NC>(
      ne, shape_1d, streams, value);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static %(inline_qualifier)s void tensor_integrate(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t *const value_coeff,
    const s_t *const grad_coeff,
    s_t *const RSTR output[NC * NS]) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template integrate<NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static %(inline_qualifier)s void tensor_integrate_contiguous(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t *const value_coeff,
    const s_t *const grad_coeff,
    s_t output[NC * NS][VS]) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template integrate_contiguous<NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static %(inline_qualifier)s void tensor_integrate_value(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const value_coeff,
    s_t *const RSTR output[NC * NS]) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template integrate_value<NC>(
      ne, shape_1d, value_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static %(inline_qualifier)s void tensor_integrate_value_contiguous(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const value_coeff,
    s_t output[NC * NS][VS]) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template integrate_value_contiguous<NC>(
      ne, shape_1d, value_coeff, output);
}

} // namespace codegen
} // namespace sfem

#endif
'''
