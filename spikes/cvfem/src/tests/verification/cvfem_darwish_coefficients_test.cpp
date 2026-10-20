#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include "kernels/microkernels/limiters/cvfem_venkata_limiter.hpp"

template <typename T> void check() {
    std::mt19937 rng(731);
    std::uniform_real_distribution<double> sample(-2., 2.);
    double worst = 0;
    for (int k = 0; k < 20000; ++k) {
        T a = T(sample(rng)), b = T(sample(rng));
        T gi = T(sample(rng)), gj = T(sample(rng));
        T vi = T(sample(rng)), vj = T(sample(rng));
        T di = T(sample(rng)), dj = T(sample(rng));
        T m = T(sample(rng)), dm = T(sample(rng));
        T scale = std::abs(a) + std::abs(b) + std::abs(gi) + std::abs(gj);
        if (k % 7 == 0) m = 0;
        if (k % 11 == 0) gi = gj = 0;
        if (k % 13 == 0) b = a;
        if (k % 17 == 0) gi = std::numeric_limits<T>::epsilon() * T(sample(rng));
        if (k % 19 == 0) b = a + std::numeric_limits<T>::epsilon() * T(sample(rng));
        const T mp = T(.5) * (m + std::abs(m)), mn = T(.5) * (m - std::abs(m));
        const T sign = m > 0 ? T(1) : m < 0 ? T(-1) : T(0);
        const T wp = T(.5) * (T(1) + sign), wn = T(.5) * (T(1) - sign);
        T oi, oj, odi, odj, ai, aj, bi, bj;
        cvfem_darwish_moukalled_inc_and_d(a,b,gi,vi,vj,di,scale,oi,odi);
        cvfem_darwish_moukalled_inc_and_d(b,a,gj,vj,vi,dj,scale,oj,odj);
        const T reference = wp*dm*oi + mp*odi + wn*dm*oj + mn*odj;
        cvfem_darwish_moukalled_inc_and_d(a,b,gi,T(0),T(0),T(1),scale,oi,ai);
        cvfem_darwish_moukalled_inc_and_d(b,a,gj,T(0),T(0),T(1),scale,oj,aj);
        cvfem_darwish_moukalled_inc_and_d(a,b,gi,T(0),T(1),T(0),scale,oi,bi);
        cvfem_darwish_moukalled_inc_and_d(b,a,gj,T(0),T(1),T(0),scale,oj,bj);
        const T candidate = (wp*oi + wn*oj)*dm + mp*ai*di + mn*aj*dj + (mp*bi - mn*bj)*(vj-vi);
        const double norm = std::max({1., std::abs(double(reference)), std::abs(double(candidate))});
        const double err = std::abs(double(reference) - double(candidate)) / norm;
        if (!std::isfinite(err) || err > (sizeof(T)==8 ? 1e-12 : 2e-5))
            throw std::runtime_error("limiter coefficient identity failed");
        worst = std::max(worst, err);
    }
    std::printf("limiter coefficient identity %zu-byte: 20000 cases, max scaled error %.9g PASS\n", sizeof(T), worst);
}


#include <cmath>
#include <cstdio>
#include <cstddef>
#include <algorithm>
#define ALIGN_BYTES 64
template<class T> constexpr size_t cvfem_hex8_vec_size=4;
#include "kernels/microkernels/hex8/cvfem_darwish_cache_experiment.hpp"
int check_lifecycle() {
    DarwishCacheExperiment<double> cache;
    int refused=0;
    auto reject=[&](auto action) {try {action();} catch(const std::runtime_error &) {++refused;return;} throw std::runtime_error("invalid cache accepted");};
    reject([&]{cache.require_ready(1);});
    reject([&]{cache.allocate(0,1);});
    cache.allocate(8,1);
    reject([&]{cache.finish_build();});
    for (auto &tile:cache.tiles) std::fill(std::begin(tile.coefficients),std::end(tile.coefficients),1.);
    cache.mark_built(0);cache.mark_built(4);cache.finish_build();cache.require_ready(1);
    reject([&]{cache.require_ready(2);});
    cache.invalidate();reject([&]{cache.require_ready(1);});
    cache.allocate(8,2);reject([&]{cache.require_ready(2);});
    cache.mark_built(0);cache.mark_built(4);
    cache.build(0)[0]=std::numeric_limits<double>::quiet_NaN();
    reject([&]{cache.finish_build();});
    cache.build(0)[0]=2.;cache.finish_build();cache.require_ready(2);
    if (refused!=7) return 1;
    std::printf("Cache lifecycle: seven refusals and rebuilt revision PASS\n");
    return 0;
}

int main() { check<double>(); check<float>(); return check_lifecycle(); }
