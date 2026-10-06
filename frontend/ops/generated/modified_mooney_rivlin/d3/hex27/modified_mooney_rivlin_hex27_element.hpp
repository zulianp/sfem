#ifndef MODIFIED_MOONEY_RIVLIN_HEX27_ELEMENT_API_HPP
#define MODIFIED_MOONEY_RIVLIN_HEX27_ELEMENT_API_HPP

#include "../proteus_hex27/modified_mooney_rivlin_proteus_hex27_element.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_hex27_energy_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int NS = 27;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[24], u_streams[25], u_streams[26], u_streams[3], u_streams[4], u_streams[5], u_streams[33], u_streams[34], u_streams[35], u_streams[72], u_streams[73], u_streams[74], u_streams[27], u_streams[28], u_streams[29], u_streams[9], u_streams[10], u_streams[11], u_streams[30], u_streams[31], u_streams[32], u_streams[6], u_streams[7], u_streams[8], u_streams[48], u_streams[49], u_streams[50], u_streams[60], u_streams[61], u_streams[62], u_streams[51], u_streams[52], u_streams[53], u_streams[69], u_streams[70], u_streams[71], u_streams[78], u_streams[79], u_streams[80], u_streams[63], u_streams[64], u_streams[65], u_streams[57], u_streams[58], u_streams[59], u_streams[66], u_streams[67], u_streams[68], u_streams[54], u_streams[55], u_streams[56], u_streams[12], u_streams[13], u_streams[14], u_streams[36], u_streams[37], u_streams[38], u_streams[15], u_streams[16], u_streams[17], u_streams[45], u_streams[46], u_streams[47], u_streams[75], u_streams[76], u_streams[77], u_streams[39], u_streams[40], u_streams[41], u_streams[21], u_streams[22], u_streams[23], u_streams[42], u_streams[43], u_streams[44], u_streams[18], u_streams[19], u_streams[20]};
  return modified_mooney_rivlin_proteus_hex27_energy_egeometry_soa<s_t, VS>(nelements, adj, det, c1, c2, kappa, ordered_u_streams, values);
}

template <typename s_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_hex27_energy_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int NS = 27;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[24], coords[25], coords[26], coords[3], coords[4], coords[5], coords[33], coords[34], coords[35], coords[72], coords[73], coords[74], coords[27], coords[28], coords[29], coords[9], coords[10], coords[11], coords[30], coords[31], coords[32], coords[6], coords[7], coords[8], coords[48], coords[49], coords[50], coords[60], coords[61], coords[62], coords[51], coords[52], coords[53], coords[69], coords[70], coords[71], coords[78], coords[79], coords[80], coords[63], coords[64], coords[65], coords[57], coords[58], coords[59], coords[66], coords[67], coords[68], coords[54], coords[55], coords[56], coords[12], coords[13], coords[14], coords[36], coords[37], coords[38], coords[15], coords[16], coords[17], coords[45], coords[46], coords[47], coords[75], coords[76], coords[77], coords[39], coords[40], coords[41], coords[21], coords[22], coords[23], coords[42], coords[43], coords[44], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[24], u_streams[25], u_streams[26], u_streams[3], u_streams[4], u_streams[5], u_streams[33], u_streams[34], u_streams[35], u_streams[72], u_streams[73], u_streams[74], u_streams[27], u_streams[28], u_streams[29], u_streams[9], u_streams[10], u_streams[11], u_streams[30], u_streams[31], u_streams[32], u_streams[6], u_streams[7], u_streams[8], u_streams[48], u_streams[49], u_streams[50], u_streams[60], u_streams[61], u_streams[62], u_streams[51], u_streams[52], u_streams[53], u_streams[69], u_streams[70], u_streams[71], u_streams[78], u_streams[79], u_streams[80], u_streams[63], u_streams[64], u_streams[65], u_streams[57], u_streams[58], u_streams[59], u_streams[66], u_streams[67], u_streams[68], u_streams[54], u_streams[55], u_streams[56], u_streams[12], u_streams[13], u_streams[14], u_streams[36], u_streams[37], u_streams[38], u_streams[15], u_streams[16], u_streams[17], u_streams[45], u_streams[46], u_streams[47], u_streams[75], u_streams[76], u_streams[77], u_streams[39], u_streams[40], u_streams[41], u_streams[21], u_streams[22], u_streams[23], u_streams[42], u_streams[43], u_streams[44], u_streams[18], u_streams[19], u_streams[20]};
  return modified_mooney_rivlin_proteus_hex27_energy_ecoords_soa<s_t, VS>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, values);
}

template <typename s_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_hex27_energy_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int NS = 27;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[24], coords[25], coords[26], coords[3], coords[4], coords[5], coords[33], coords[34], coords[35], coords[72], coords[73], coords[74], coords[27], coords[28], coords[29], coords[9], coords[10], coords[11], coords[30], coords[31], coords[32], coords[6], coords[7], coords[8], coords[48], coords[49], coords[50], coords[60], coords[61], coords[62], coords[51], coords[52], coords[53], coords[69], coords[70], coords[71], coords[78], coords[79], coords[80], coords[63], coords[64], coords[65], coords[57], coords[58], coords[59], coords[66], coords[67], coords[68], coords[54], coords[55], coords[56], coords[12], coords[13], coords[14], coords[36], coords[37], coords[38], coords[15], coords[16], coords[17], coords[45], coords[46], coords[47], coords[75], coords[76], coords[77], coords[39], coords[40], coords[41], coords[21], coords[22], coords[23], coords[42], coords[43], coords[44], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[24], u_streams[25], u_streams[26], u_streams[3], u_streams[4], u_streams[5], u_streams[33], u_streams[34], u_streams[35], u_streams[72], u_streams[73], u_streams[74], u_streams[27], u_streams[28], u_streams[29], u_streams[9], u_streams[10], u_streams[11], u_streams[30], u_streams[31], u_streams[32], u_streams[6], u_streams[7], u_streams[8], u_streams[48], u_streams[49], u_streams[50], u_streams[60], u_streams[61], u_streams[62], u_streams[51], u_streams[52], u_streams[53], u_streams[69], u_streams[70], u_streams[71], u_streams[78], u_streams[79], u_streams[80], u_streams[63], u_streams[64], u_streams[65], u_streams[57], u_streams[58], u_streams[59], u_streams[66], u_streams[67], u_streams[68], u_streams[54], u_streams[55], u_streams[56], u_streams[12], u_streams[13], u_streams[14], u_streams[36], u_streams[37], u_streams[38], u_streams[15], u_streams[16], u_streams[17], u_streams[45], u_streams[46], u_streams[47], u_streams[75], u_streams[76], u_streams[77], u_streams[39], u_streams[40], u_streams[41], u_streams[21], u_streams[22], u_streams[23], u_streams[42], u_streams[43], u_streams[44], u_streams[18], u_streams[19], u_streams[20]};
  return modified_mooney_rivlin_proteus_hex27_energy_esoa<s_t, VS>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, values);
}

template <typename s_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_hex27_gradient_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 27;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[24], u_streams[25], u_streams[26], u_streams[3], u_streams[4], u_streams[5], u_streams[33], u_streams[34], u_streams[35], u_streams[72], u_streams[73], u_streams[74], u_streams[27], u_streams[28], u_streams[29], u_streams[9], u_streams[10], u_streams[11], u_streams[30], u_streams[31], u_streams[32], u_streams[6], u_streams[7], u_streams[8], u_streams[48], u_streams[49], u_streams[50], u_streams[60], u_streams[61], u_streams[62], u_streams[51], u_streams[52], u_streams[53], u_streams[69], u_streams[70], u_streams[71], u_streams[78], u_streams[79], u_streams[80], u_streams[63], u_streams[64], u_streams[65], u_streams[57], u_streams[58], u_streams[59], u_streams[66], u_streams[67], u_streams[68], u_streams[54], u_streams[55], u_streams[56], u_streams[12], u_streams[13], u_streams[14], u_streams[36], u_streams[37], u_streams[38], u_streams[15], u_streams[16], u_streams[17], u_streams[45], u_streams[46], u_streams[47], u_streams[75], u_streams[76], u_streams[77], u_streams[39], u_streams[40], u_streams[41], u_streams[21], u_streams[22], u_streams[23], u_streams[42], u_streams[43], u_streams[44], u_streams[18], u_streams[19], u_streams[20]};
  s_t *const ordered_out_streams[NDOFS] = {out_streams[0], out_streams[1], out_streams[2], out_streams[24], out_streams[25], out_streams[26], out_streams[3], out_streams[4], out_streams[5], out_streams[33], out_streams[34], out_streams[35], out_streams[72], out_streams[73], out_streams[74], out_streams[27], out_streams[28], out_streams[29], out_streams[9], out_streams[10], out_streams[11], out_streams[30], out_streams[31], out_streams[32], out_streams[6], out_streams[7], out_streams[8], out_streams[48], out_streams[49], out_streams[50], out_streams[60], out_streams[61], out_streams[62], out_streams[51], out_streams[52], out_streams[53], out_streams[69], out_streams[70], out_streams[71], out_streams[78], out_streams[79], out_streams[80], out_streams[63], out_streams[64], out_streams[65], out_streams[57], out_streams[58], out_streams[59], out_streams[66], out_streams[67], out_streams[68], out_streams[54], out_streams[55], out_streams[56], out_streams[12], out_streams[13], out_streams[14], out_streams[36], out_streams[37], out_streams[38], out_streams[15], out_streams[16], out_streams[17], out_streams[45], out_streams[46], out_streams[47], out_streams[75], out_streams[76], out_streams[77], out_streams[39], out_streams[40], out_streams[41], out_streams[21], out_streams[22], out_streams[23], out_streams[42], out_streams[43], out_streams[44], out_streams[18], out_streams[19], out_streams[20]};
  return modified_mooney_rivlin_proteus_hex27_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, c1, c2, kappa, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_hex27_gradient_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 27;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[24], coords[25], coords[26], coords[3], coords[4], coords[5], coords[33], coords[34], coords[35], coords[72], coords[73], coords[74], coords[27], coords[28], coords[29], coords[9], coords[10], coords[11], coords[30], coords[31], coords[32], coords[6], coords[7], coords[8], coords[48], coords[49], coords[50], coords[60], coords[61], coords[62], coords[51], coords[52], coords[53], coords[69], coords[70], coords[71], coords[78], coords[79], coords[80], coords[63], coords[64], coords[65], coords[57], coords[58], coords[59], coords[66], coords[67], coords[68], coords[54], coords[55], coords[56], coords[12], coords[13], coords[14], coords[36], coords[37], coords[38], coords[15], coords[16], coords[17], coords[45], coords[46], coords[47], coords[75], coords[76], coords[77], coords[39], coords[40], coords[41], coords[21], coords[22], coords[23], coords[42], coords[43], coords[44], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[24], u_streams[25], u_streams[26], u_streams[3], u_streams[4], u_streams[5], u_streams[33], u_streams[34], u_streams[35], u_streams[72], u_streams[73], u_streams[74], u_streams[27], u_streams[28], u_streams[29], u_streams[9], u_streams[10], u_streams[11], u_streams[30], u_streams[31], u_streams[32], u_streams[6], u_streams[7], u_streams[8], u_streams[48], u_streams[49], u_streams[50], u_streams[60], u_streams[61], u_streams[62], u_streams[51], u_streams[52], u_streams[53], u_streams[69], u_streams[70], u_streams[71], u_streams[78], u_streams[79], u_streams[80], u_streams[63], u_streams[64], u_streams[65], u_streams[57], u_streams[58], u_streams[59], u_streams[66], u_streams[67], u_streams[68], u_streams[54], u_streams[55], u_streams[56], u_streams[12], u_streams[13], u_streams[14], u_streams[36], u_streams[37], u_streams[38], u_streams[15], u_streams[16], u_streams[17], u_streams[45], u_streams[46], u_streams[47], u_streams[75], u_streams[76], u_streams[77], u_streams[39], u_streams[40], u_streams[41], u_streams[21], u_streams[22], u_streams[23], u_streams[42], u_streams[43], u_streams[44], u_streams[18], u_streams[19], u_streams[20]};
  s_t *const ordered_out_streams[NDOFS] = {out_streams[0], out_streams[1], out_streams[2], out_streams[24], out_streams[25], out_streams[26], out_streams[3], out_streams[4], out_streams[5], out_streams[33], out_streams[34], out_streams[35], out_streams[72], out_streams[73], out_streams[74], out_streams[27], out_streams[28], out_streams[29], out_streams[9], out_streams[10], out_streams[11], out_streams[30], out_streams[31], out_streams[32], out_streams[6], out_streams[7], out_streams[8], out_streams[48], out_streams[49], out_streams[50], out_streams[60], out_streams[61], out_streams[62], out_streams[51], out_streams[52], out_streams[53], out_streams[69], out_streams[70], out_streams[71], out_streams[78], out_streams[79], out_streams[80], out_streams[63], out_streams[64], out_streams[65], out_streams[57], out_streams[58], out_streams[59], out_streams[66], out_streams[67], out_streams[68], out_streams[54], out_streams[55], out_streams[56], out_streams[12], out_streams[13], out_streams[14], out_streams[36], out_streams[37], out_streams[38], out_streams[15], out_streams[16], out_streams[17], out_streams[45], out_streams[46], out_streams[47], out_streams[75], out_streams[76], out_streams[77], out_streams[39], out_streams[40], out_streams[41], out_streams[21], out_streams[22], out_streams[23], out_streams[42], out_streams[43], out_streams[44], out_streams[18], out_streams[19], out_streams[20]};
  return modified_mooney_rivlin_proteus_hex27_gradient_ecoords_soa<s_t, VS>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_hex27_gradient_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 27;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[24], coords[25], coords[26], coords[3], coords[4], coords[5], coords[33], coords[34], coords[35], coords[72], coords[73], coords[74], coords[27], coords[28], coords[29], coords[9], coords[10], coords[11], coords[30], coords[31], coords[32], coords[6], coords[7], coords[8], coords[48], coords[49], coords[50], coords[60], coords[61], coords[62], coords[51], coords[52], coords[53], coords[69], coords[70], coords[71], coords[78], coords[79], coords[80], coords[63], coords[64], coords[65], coords[57], coords[58], coords[59], coords[66], coords[67], coords[68], coords[54], coords[55], coords[56], coords[12], coords[13], coords[14], coords[36], coords[37], coords[38], coords[15], coords[16], coords[17], coords[45], coords[46], coords[47], coords[75], coords[76], coords[77], coords[39], coords[40], coords[41], coords[21], coords[22], coords[23], coords[42], coords[43], coords[44], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[24], u_streams[25], u_streams[26], u_streams[3], u_streams[4], u_streams[5], u_streams[33], u_streams[34], u_streams[35], u_streams[72], u_streams[73], u_streams[74], u_streams[27], u_streams[28], u_streams[29], u_streams[9], u_streams[10], u_streams[11], u_streams[30], u_streams[31], u_streams[32], u_streams[6], u_streams[7], u_streams[8], u_streams[48], u_streams[49], u_streams[50], u_streams[60], u_streams[61], u_streams[62], u_streams[51], u_streams[52], u_streams[53], u_streams[69], u_streams[70], u_streams[71], u_streams[78], u_streams[79], u_streams[80], u_streams[63], u_streams[64], u_streams[65], u_streams[57], u_streams[58], u_streams[59], u_streams[66], u_streams[67], u_streams[68], u_streams[54], u_streams[55], u_streams[56], u_streams[12], u_streams[13], u_streams[14], u_streams[36], u_streams[37], u_streams[38], u_streams[15], u_streams[16], u_streams[17], u_streams[45], u_streams[46], u_streams[47], u_streams[75], u_streams[76], u_streams[77], u_streams[39], u_streams[40], u_streams[41], u_streams[21], u_streams[22], u_streams[23], u_streams[42], u_streams[43], u_streams[44], u_streams[18], u_streams[19], u_streams[20]};
  s_t *const ordered_out_streams[NDOFS] = {out_streams[0], out_streams[1], out_streams[2], out_streams[24], out_streams[25], out_streams[26], out_streams[3], out_streams[4], out_streams[5], out_streams[33], out_streams[34], out_streams[35], out_streams[72], out_streams[73], out_streams[74], out_streams[27], out_streams[28], out_streams[29], out_streams[9], out_streams[10], out_streams[11], out_streams[30], out_streams[31], out_streams[32], out_streams[6], out_streams[7], out_streams[8], out_streams[48], out_streams[49], out_streams[50], out_streams[60], out_streams[61], out_streams[62], out_streams[51], out_streams[52], out_streams[53], out_streams[69], out_streams[70], out_streams[71], out_streams[78], out_streams[79], out_streams[80], out_streams[63], out_streams[64], out_streams[65], out_streams[57], out_streams[58], out_streams[59], out_streams[66], out_streams[67], out_streams[68], out_streams[54], out_streams[55], out_streams[56], out_streams[12], out_streams[13], out_streams[14], out_streams[36], out_streams[37], out_streams[38], out_streams[15], out_streams[16], out_streams[17], out_streams[45], out_streams[46], out_streams[47], out_streams[75], out_streams[76], out_streams[77], out_streams[39], out_streams[40], out_streams[41], out_streams[21], out_streams[22], out_streams[23], out_streams[42], out_streams[43], out_streams[44], out_streams[18], out_streams[19], out_streams[20]};
  return modified_mooney_rivlin_proteus_hex27_gradient_esoa<s_t, VS>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_hex27_hessian_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 27;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[24], u_streams[25], u_streams[26], u_streams[3], u_streams[4], u_streams[5], u_streams[33], u_streams[34], u_streams[35], u_streams[72], u_streams[73], u_streams[74], u_streams[27], u_streams[28], u_streams[29], u_streams[9], u_streams[10], u_streams[11], u_streams[30], u_streams[31], u_streams[32], u_streams[6], u_streams[7], u_streams[8], u_streams[48], u_streams[49], u_streams[50], u_streams[60], u_streams[61], u_streams[62], u_streams[51], u_streams[52], u_streams[53], u_streams[69], u_streams[70], u_streams[71], u_streams[78], u_streams[79], u_streams[80], u_streams[63], u_streams[64], u_streams[65], u_streams[57], u_streams[58], u_streams[59], u_streams[66], u_streams[67], u_streams[68], u_streams[54], u_streams[55], u_streams[56], u_streams[12], u_streams[13], u_streams[14], u_streams[36], u_streams[37], u_streams[38], u_streams[15], u_streams[16], u_streams[17], u_streams[45], u_streams[46], u_streams[47], u_streams[75], u_streams[76], u_streams[77], u_streams[39], u_streams[40], u_streams[41], u_streams[21], u_streams[22], u_streams[23], u_streams[42], u_streams[43], u_streams[44], u_streams[18], u_streams[19], u_streams[20]};
  s_t *ordered_matrix_streams[NDOFS * NDOFS];
  for (int row_shape = 0; row_shape < NS; ++row_shape) {
    const int source_row_shape = SHAPE_ORDER[row_shape];
    for (int row_component = 0; row_component < NC; ++row_component) {
      const int row = row_shape * NC + row_component;
      const int source_row = source_row_shape * NC + row_component;
      for (int col_shape = 0; col_shape < NS; ++col_shape) {
        const int source_col_shape = SHAPE_ORDER[col_shape];
        for (int col_component = 0; col_component < NC; ++col_component) {
          const int col = col_shape * NC + col_component;
          const int source_col = source_col_shape * NC + col_component;
          ordered_matrix_streams[row * NDOFS + col] = matrix_streams[source_row * NDOFS + source_col];
        }
      }
    }
  }
  return modified_mooney_rivlin_proteus_hex27_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, c1, c2, kappa, ordered_u_streams, ordered_matrix_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_hex27_hessian_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 27;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6};
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[24], coords[25], coords[26], coords[3], coords[4], coords[5], coords[33], coords[34], coords[35], coords[72], coords[73], coords[74], coords[27], coords[28], coords[29], coords[9], coords[10], coords[11], coords[30], coords[31], coords[32], coords[6], coords[7], coords[8], coords[48], coords[49], coords[50], coords[60], coords[61], coords[62], coords[51], coords[52], coords[53], coords[69], coords[70], coords[71], coords[78], coords[79], coords[80], coords[63], coords[64], coords[65], coords[57], coords[58], coords[59], coords[66], coords[67], coords[68], coords[54], coords[55], coords[56], coords[12], coords[13], coords[14], coords[36], coords[37], coords[38], coords[15], coords[16], coords[17], coords[45], coords[46], coords[47], coords[75], coords[76], coords[77], coords[39], coords[40], coords[41], coords[21], coords[22], coords[23], coords[42], coords[43], coords[44], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[24], u_streams[25], u_streams[26], u_streams[3], u_streams[4], u_streams[5], u_streams[33], u_streams[34], u_streams[35], u_streams[72], u_streams[73], u_streams[74], u_streams[27], u_streams[28], u_streams[29], u_streams[9], u_streams[10], u_streams[11], u_streams[30], u_streams[31], u_streams[32], u_streams[6], u_streams[7], u_streams[8], u_streams[48], u_streams[49], u_streams[50], u_streams[60], u_streams[61], u_streams[62], u_streams[51], u_streams[52], u_streams[53], u_streams[69], u_streams[70], u_streams[71], u_streams[78], u_streams[79], u_streams[80], u_streams[63], u_streams[64], u_streams[65], u_streams[57], u_streams[58], u_streams[59], u_streams[66], u_streams[67], u_streams[68], u_streams[54], u_streams[55], u_streams[56], u_streams[12], u_streams[13], u_streams[14], u_streams[36], u_streams[37], u_streams[38], u_streams[15], u_streams[16], u_streams[17], u_streams[45], u_streams[46], u_streams[47], u_streams[75], u_streams[76], u_streams[77], u_streams[39], u_streams[40], u_streams[41], u_streams[21], u_streams[22], u_streams[23], u_streams[42], u_streams[43], u_streams[44], u_streams[18], u_streams[19], u_streams[20]};
  s_t *ordered_matrix_streams[NDOFS * NDOFS];
  for (int row_shape = 0; row_shape < NS; ++row_shape) {
    const int source_row_shape = SHAPE_ORDER[row_shape];
    for (int row_component = 0; row_component < NC; ++row_component) {
      const int row = row_shape * NC + row_component;
      const int source_row = source_row_shape * NC + row_component;
      for (int col_shape = 0; col_shape < NS; ++col_shape) {
        const int source_col_shape = SHAPE_ORDER[col_shape];
        for (int col_component = 0; col_component < NC; ++col_component) {
          const int col = col_shape * NC + col_component;
          const int source_col = source_col_shape * NC + col_component;
          ordered_matrix_streams[row * NDOFS + col] = matrix_streams[source_row * NDOFS + source_col];
        }
      }
    }
  }
  return modified_mooney_rivlin_proteus_hex27_hessian_ecoords_soa<s_t, VS>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, ordered_matrix_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_hex27_hessian_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 27;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6};
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[24], coords[25], coords[26], coords[3], coords[4], coords[5], coords[33], coords[34], coords[35], coords[72], coords[73], coords[74], coords[27], coords[28], coords[29], coords[9], coords[10], coords[11], coords[30], coords[31], coords[32], coords[6], coords[7], coords[8], coords[48], coords[49], coords[50], coords[60], coords[61], coords[62], coords[51], coords[52], coords[53], coords[69], coords[70], coords[71], coords[78], coords[79], coords[80], coords[63], coords[64], coords[65], coords[57], coords[58], coords[59], coords[66], coords[67], coords[68], coords[54], coords[55], coords[56], coords[12], coords[13], coords[14], coords[36], coords[37], coords[38], coords[15], coords[16], coords[17], coords[45], coords[46], coords[47], coords[75], coords[76], coords[77], coords[39], coords[40], coords[41], coords[21], coords[22], coords[23], coords[42], coords[43], coords[44], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[24], u_streams[25], u_streams[26], u_streams[3], u_streams[4], u_streams[5], u_streams[33], u_streams[34], u_streams[35], u_streams[72], u_streams[73], u_streams[74], u_streams[27], u_streams[28], u_streams[29], u_streams[9], u_streams[10], u_streams[11], u_streams[30], u_streams[31], u_streams[32], u_streams[6], u_streams[7], u_streams[8], u_streams[48], u_streams[49], u_streams[50], u_streams[60], u_streams[61], u_streams[62], u_streams[51], u_streams[52], u_streams[53], u_streams[69], u_streams[70], u_streams[71], u_streams[78], u_streams[79], u_streams[80], u_streams[63], u_streams[64], u_streams[65], u_streams[57], u_streams[58], u_streams[59], u_streams[66], u_streams[67], u_streams[68], u_streams[54], u_streams[55], u_streams[56], u_streams[12], u_streams[13], u_streams[14], u_streams[36], u_streams[37], u_streams[38], u_streams[15], u_streams[16], u_streams[17], u_streams[45], u_streams[46], u_streams[47], u_streams[75], u_streams[76], u_streams[77], u_streams[39], u_streams[40], u_streams[41], u_streams[21], u_streams[22], u_streams[23], u_streams[42], u_streams[43], u_streams[44], u_streams[18], u_streams[19], u_streams[20]};
  s_t *ordered_matrix_streams[NDOFS * NDOFS];
  for (int row_shape = 0; row_shape < NS; ++row_shape) {
    const int source_row_shape = SHAPE_ORDER[row_shape];
    for (int row_component = 0; row_component < NC; ++row_component) {
      const int row = row_shape * NC + row_component;
      const int source_row = source_row_shape * NC + row_component;
      for (int col_shape = 0; col_shape < NS; ++col_shape) {
        const int source_col_shape = SHAPE_ORDER[col_shape];
        for (int col_component = 0; col_component < NC; ++col_component) {
          const int col = col_shape * NC + col_component;
          const int source_col = source_col_shape * NC + col_component;
          ordered_matrix_streams[row * NDOFS + col] = matrix_streams[source_row * NDOFS + source_col];
        }
      }
    }
  }
  return modified_mooney_rivlin_proteus_hex27_hessian_esoa<s_t, VS>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, ordered_matrix_streams);
}

} // namespace codegen
} // namespace sfem

#endif
