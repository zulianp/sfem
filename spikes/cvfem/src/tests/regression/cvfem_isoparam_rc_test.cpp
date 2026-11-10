#include "frontend/staging/cvfem_hex8_best_common.hpp"
#include "kernels/standard/isoparametric/cvfem_hex8_best_atomic_isoparam.hpp"

// A physical linear pressure must have zero Rhie-Chow correction even when
// opposite edges differ. The old affine-edge shortcut fails this check.
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int result = 0;
    {
        MeshData d;
        d.mesh = smesh::Mesh::create_hex8_cube(smesh::Communicator::self(), 1, 1, 1, 0, 0, 0, 1, 1, 1);
        d.nnodes=d.mesh->n_nodes(); d.nelements=d.mesh->n_elements(0);
        d.elems=d.mesh->elements(0)->data(); d.points=d.mesh->points()->data();
        if(d.nelements!=1 || d.nnodes!=8) result=1;
        else {
            d.points[0][d.elems[6][0]] += geom_t(.25);
            precompute_affine_geometry(d);
            fill_fields(d);
            d.pgx.assign(8,1); d.pgy.assign(8,2); d.pgz.assign(8,3);
            for(int i=0;i<8;++i) d.p[i]=d.points[0][i]+2.*d.points[1][i]+3.*d.points[2][i];
            const auto run=[&](scalar_t scale) {
                d.rhie_chow_scale=scale;
                apply_residual_atomic_isoparam(d.adj_ptr,d.det_ptr,d.elems,d.face_mask.data(),
                    d.nelements,d.nnodes,d.p.data(),d.pgx.data(),d.pgy.data(),d.pgz.data(),d.points,
                    d.qgx.data(),d.qgy.data(),d.qgz.data(),d.rc.data(),d.rx.data(),d.ry.data(),d.rz.data(),
                    d.ux.data(),d.uy.data(),d.uz.data(),cvfem_hex8_extras_of(d),scalar_t(1.3),scalar_t(.027));
                std::vector<scalar_t> v;
                for(const auto *a:{&d.rx,&d.ry,&d.rz,&d.rc}) v.insert(v.end(),a->begin(),a->end());
                return v;
            };
            const auto bare=run(0), corrected=run(1);
            double good=0, bad=0;
            for(size_t i=0;i<bare.size();++i) {
                if(!std::isfinite(bare[i]) || !std::isfinite(corrected[i])) result=1;
                good=std::max(good,std::abs(bare[i]-corrected[i]));
            }
            Hex8ExtraScratch old;
            old.load(d.elems,d.points,d.face_mask.data(),d.pgx.data(),d.pgy.data(),d.pgz.data(),
                d.qgx.data(),d.qgy.data(),d.qgz.data(),d.ux.data(),d.uy.data(),d.uz.data(),
                d.adj_ptr,d.det_ptr,cvfem_hex8_extras_of(d),0,true);
            scalar_t ux[8],uy[8],uz[8],p[8],r[32];
            gather_element_fields(d.elems,d.ux.data(),d.uy.data(),d.uz.data(),d.p.data(),0,ux,uy,uz,p);
            cvfem_hex8_ns_upwind_residual_isoparam(scalar_t(1.3),scalar_t(.027),old.x,old.y,old.z,
                ux,uy,uz,p,r,old.rc);
            for(int a=0;a<8;++a)for(int c=0;c<4;++c) {
                if(!std::isfinite(r[4*a+c])) result=1;
                bad=std::max(bad,std::abs(r[4*a+c]-bare[8*c+d.elems[a][0]]));
            }
            std::printf("linear pressure on distorted HEX8: corrected=%.12g old_affine_edges=%.12g\n",good,bad);
            if(good>1e-12 || bad<1e-5) result=1;
        }
    }
    MPI_Finalize();
    return result;
}
