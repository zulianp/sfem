#include "frontend/staging/cvfem_hex8_best_common.hpp"
#include "kernels/standard/isoparametric/cvfem_hex8_best_atomic_isoparam.hpp"
#include "frontend/staging/cvfem_hex8_packed_launch.hpp"
#include <array>
#include <stdexcept>
#include <string>

struct UnstructuredRun {
    MeshData d;
    PackedData p;
    bool packed = false;
    std::vector<scalar_t> grad_buffer, out;
    std::array<std::vector<scalar_t>, 3> direction;
    std::array<std::vector<scalar_t>, 4> state;
    double packing_seconds = 0, ordering_seconds = 0;
};

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void load_run(UnstructuredRun &r, const std::string &path, int pack) {
    r.d.mesh = smesh::Mesh::create_from_file(smesh::Communicator::self(), smesh::Path(path));
    require(r.d.mesh && r.d.mesh->n_nodes() > 0 && r.d.mesh->element_type(0) == smesh::HEX8,
            "expected a nonempty HEX8 mesh");
    double t = wall_time();
    smesh::SFC::create_from_env()->reorder(*r.d.mesh);
    r.ordering_seconds = wall_time()-t;
    r.packed = pack > 0;
    if (r.packed) { t=wall_time(); r.p=make_packed(r.d.mesh,pack); r.packing_seconds=wall_time()-t; }
    auto &d=r.d;
    d.nnodes=d.mesh->n_nodes(); d.nelements=d.mesh->n_elements(0);
    d.points=d.mesh->points()->data(); d.elems=d.mesh->elements(0)->data();
    fill_fields(d);
    d.rhie_chow_scale=1;
    r.out.resize(size_t(d.nnodes)*4);
    for (auto &v:r.direction) v.resize(r.out.size());
}

static void gradient(UnstructuredRun &r, const scalar_t *src, int stride,
                     std::vector<scalar_t> &gx, std::vector<scalar_t> &gy, std::vector<scalar_t> &gz) {
    if(r.packed) cvfem_hex8_assemble_nodal_grads_packed(r.d,r.p,1,src,stride,gx,gy,gz,r.grad_buffer);
    else cvfem_hex8_assemble_nodal_grads_atomic(r.d,1,src,stride,gx,gy,gz);
}

static void state_gradient(UnstructuredRun &r) {
    gradient(r,r.d.p.data(),1,r.d.pgx,r.d.pgy,r.d.pgz);
}

static void verify_linear_gradient(UnstructuredRun &r) {
    std::vector<scalar_t> pressure(size_t(r.d.nnodes)), gx, gy, gz;
    for(ptrdiff_t i=0;i<r.d.nnodes;++i)
        pressure[i]=r.d.points[0][i]+2.*r.d.points[1][i]+3.*r.d.points[2][i];
    gradient(r,pressure.data(),1,gx,gy,gz);
    double error=0;
    for(ptrdiff_t i=0;i<r.d.nnodes;++i) {
        require(std::isfinite(gx[i])&&std::isfinite(gy[i])&&std::isfinite(gz[i]),"nonfinite reconstructed gradient");
        error=std::max({error,std::abs(gx[i]-1),std::abs(gy[i]-2),std::abs(gz[i]-3)});
    }
    std::printf("linear_gradient,layout=%s,error=%.12g\n",r.packed?"packed":"atomic",error);
    require(error<1e-11,"physical linear gradient is not reproduced");
}

static void apply(UnstructuredRun &r, bool jacobian, int direction=0, bool omit_qgrad=false) {
    auto &d=r.d;
    constexpr scalar_t rho=1.3, mu=.027;
    if(jacobian) {
        const auto &v=r.direction[direction];
        gradient(r,v.data()+3,4,d.qgx,d.qgy,d.qgz);
        if(omit_qgrad) { d.qgx.clear();d.qgy.clear();d.qgz.clear(); }
        if(r.packed) apply_jacobian_action_packed(d,r.p,rho,mu,GeomKind::Isoparam,v.data(),r.out.data());
        else apply_jacobian_action_atomic_isoparam(d.adj_ptr,d.det_ptr,d.elems,d.face_mask.data(),
            d.nelements,d.nnodes,d.p.data(),d.pgx.data(),d.pgy.data(),d.pgz.data(),d.points,
            d.qgx.data(),d.qgy.data(),d.qgz.data(),d.ux.data(),d.uy.data(),d.uz.data(),
            cvfem_hex8_extras_of(d),rho,mu,v.data(),r.out.data());
    } else {
        state_gradient(r);
        if(r.packed) apply_residual_packed(d,r.p,rho,mu,GeomKind::Isoparam);
        else apply_residual_atomic_isoparam(d.adj_ptr,d.det_ptr,d.elems,d.face_mask.data(),
            d.nelements,d.nnodes,d.p.data(),d.pgx.data(),d.pgy.data(),d.pgz.data(),d.points,
            d.qgx.data(),d.qgy.data(),d.qgz.data(),d.rc.data(),d.rx.data(),d.ry.data(),d.rz.data(),
            d.ux.data(),d.uy.data(),d.uz.data(),cvfem_hex8_extras_of(d),rho,mu);
        // Result packing is checked outside timing; the residual writes its native SoA.
    }
}

static std::vector<scalar_t> output(const UnstructuredRun &r, bool jacobian) {
    if(jacobian) return r.out;
    std::vector<scalar_t> v(size_t(r.d.nnodes)*4);
    for(ptrdiff_t i=0;i<r.d.nnodes;++i) {
        v[4*i]=r.d.rx[i];v[4*i+1]=r.d.ry[i];v[4*i+2]=r.d.rz[i];v[4*i+3]=r.d.rc[i];
    }
    return v;
}

static double relative_error(const std::vector<scalar_t> &a, const std::vector<scalar_t> &b) {
    require(a.size()==b.size()&&!a.empty(),"incompatible vectors");
    std::array<double,4> error{},scale{};
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite operator output");
        error[i%4]=std::max(error[i%4],std::abs(a[i]-b[i]));
        scale[i%4]=std::max(scale[i%4],std::max(std::abs(a[i]),std::abs(b[i])));
    }
    double worst=0;
    for(int c=0;c<4;++c) {
        require(scale[c]>0,"zero equation reference");
        worst=std::max(worst,error[c]/scale[c]);
    }
    return worst;
}

static void conservation(const std::vector<scalar_t> &v) {
    std::array<long double,4> sum{},scale{};
    for(size_t i=0;i<v.size();++i) { require(std::isfinite(v[i]),"nonfinite flux");sum[i%4]+=v[i];scale[i%4]+=std::abs(v[i]); }
    for(int c=0;c<4;++c) require(std::abs(sum[c])<=1e-11L*scale[c],"interior face cancellation failed");
}

static void set_step(UnstructuredRun &r, double step, int k) {
    auto &d=r.d;std::vector<scalar_t>* fields[]={&d.ux,&d.uy,&d.uz,&d.p};
    for(int c=0;c<4;++c) for(ptrdiff_t i=0;i<d.nnodes;++i)
        (*fields[c])[i]=r.state[c][i]+step*r.direction[k][4*i+c];
    ++d.state_stamp;
}

static void verify_fd(UnstructuredRun &r) {
    for(int k=0;k<3;++k) {
        set_step(r,0,k);state_gradient(r);apply(r,true,k);auto exact=r.out;
        apply(r,true,k,true);auto omitted=r.out;
        for(double eps:{1e-6,1e-7}) {
            set_step(r,eps,k);apply(r,false);auto plus=output(r,false);
            set_step(r,-eps,k);apply(r,false);auto minus=output(r,false);
            for(size_t i=0;i<plus.size();++i) plus[i]=(plus[i]-minus[i])/(2*eps);
            double err=relative_error(exact,plus),control=relative_error(omitted,plus);
            std::printf("fd,%s,direction=%d,eps=%.1e,error=%.12g,no_qgrad=%.12g\n",
                        r.packed?"packed":"atomic",k,eps,err,control);
            require(err<1e-6,"composed isoparametric finite difference failed");
            require(control>1e-5,"direction-gradient omission control was not detected");
        }
    }
    set_step(r,0,0);state_gradient(r);
}

static void geometry_check(const MeshData &d) {
    double smallest=std::numeric_limits<double>::infinity();
    for(ptrdiff_t e=0;e<d.nelements;++e) {
        scalar_t x[8],y[8],z[8];
        for(int a=0;a<8;++a) {const auto i=d.elems[a][e];x[a]=d.points[0][i];y[a]=d.points[1][i];z[a]=d.points[2][i];}
        for(double qx:{0.,.5,1.}) for(double qy:{0.,.5,1.}) for(double qz:{0.,.5,1.}) {
            scalar_t adj[9],det;cvfem_hex8_geom_at(x,y,z,qx,qy,qz,adj,&det);
            require(std::isfinite(det)&&det>0,"invalid element Jacobian");smallest=std::min(smallest,det);
        }
    }
    std::printf("geometry,min_sampled_det=%.12g\n",smallest);
}

static size_t mesh_bytes(const UnstructuredRun &r) {
    return r.d.mesh->points()->nbytes()+r.d.mesh->elements(0)->nbytes();
}

static int run(int argc,char **argv) {
    std::string path;int pack=1024,repeat=3,warmup=3,blocks=12;bool verify_only=false;
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--verify-only") {verify_only=true;continue;}
        require(i+1<argc,"missing argument value");
        std::string value=argv[++i];
        if(arg=="--mesh") path=value;
        else {
            size_t pos=0;int n=std::stoi(value,&pos);require(pos==value.size()&&n>0,"invalid positive integer");
            if(arg=="--pack-size")pack=n;else if(arg=="--repeat")repeat=n;
            else if(arg=="--warmup")warmup=n;else if(arg=="--blocks")blocks=n;
            else throw std::runtime_error("unknown argument: "+arg);
        }
    }
    require(!path.empty()&&pack<=8192&&blocks<=100&&repeat<=1000&&warmup<=100,"unsupported run scope");
    int threads=1;
#pragma omp parallel
    {
#pragma omp single
        threads=omp_get_num_threads();
    }
    UnstructuredRun atomic,packed;
    load_run(atomic,path,0);load_run(packed,path,pack);
    require(atomic.d.nnodes==packed.d.nnodes&&atomic.d.nelements==packed.d.nelements,"different meshes");
    geometry_check(atomic.d);
    verify_linear_gradient(atomic);verify_linear_gradient(packed);
    const auto *map=packed.p.packed->node_map()->data();
    std::vector<unsigned char> seen(size_t(atomic.d.nnodes),0);
    for(ptrdiff_t i=0;i<atomic.d.nnodes;++i) {
        const auto j=map[i];require(j>=0&&j<packed.d.nnodes&&!seen[j],"invalid node permutation");seen[j]=1;
        for(int c=0;c<3;++c)require(atomic.d.points[c][i]==packed.d.points[c][j],"coordinate permutation mismatch");
        const double x=atomic.d.points[0][i],y=atomic.d.points[1][i],z=atomic.d.points[2][i];
        atomic.d.p[i]=.3*std::sin(2*x+3*y+5*z)+.17*x*y;
        std::vector<scalar_t>* a[]={&atomic.d.ux,&atomic.d.uy,&atomic.d.uz,&atomic.d.p};
        std::vector<scalar_t>* b[]={&packed.d.ux,&packed.d.uy,&packed.d.uz,&packed.d.p};
        for(int c=0;c<4;++c)(*b[c])[j]=(*a[c])[i];
        for(int k=0;k<3;++k)for(int c=0;c<4;++c) {
            const double value=.11*std::sin((c+1.3)*x+(k+2.1)*y+(c+k+.7)*z+.3)+.03*(c+1);
            atomic.direction[k][4*i+c]=value;packed.direction[k][4*j+c]=value;
        }
    }
    for(auto *r:{&atomic,&packed}) {
        r->state={r->d.ux,r->d.uy,r->d.uz,r->d.p};
        const auto cfg=cvfem_hex8_rc_config_for(r->d);
        require(cfg.scale==1&&cfg.tau.u2_scale==1&&cfg.tau.inv_dt_a0==0,"requires the steady combined RC coefficient");
        state_gradient(*r);
    }
    std::printf("scope,first_order_RC,isoparametric,steady_interior,rho=1.3,mu=0.027,threads=%d,elements=%td,nodes=%td,pack=%d\n",
        threads,atomic.d.nelements,atomic.d.nnodes,pack);
    verify_fd(atomic);verify_fd(packed);
    std::array<std::array<std::vector<scalar_t>,3>,2> reference;
    for(int op=0;op<2;++op)for(int k=0;k<(op?3:1);++k) {
        apply(atomic,op,k);apply(packed,op,k);
        auto a=output(atomic,op),b=output(packed,op);conservation(a);conservation(b);
        std::vector<scalar_t> ordered(a.size());
        for(ptrdiff_t i=0;i<atomic.d.nnodes;++i)for(int c=0;c<4;++c)ordered[4*i+c]=b[4*map[i]+c];
        double err=relative_error(a,ordered);
        std::printf("agreement,op=%d,direction=%d,error=%.12g\n",op,k,err);
        require(err<1e-11,"packed/atomic operator mismatch");reference[op][k]=a;
    }
    // Count resident numeric arrays, not allocator overhead, stacks, or peak RSS.
    for(auto *r:{&atomic,&packed}) {
        size_t fields=0, harness=0;
        for(const auto *v:{&r->d.ux,&r->d.uy,&r->d.uz,&r->d.p,&r->d.rx,&r->d.ry,&r->d.rz,&r->d.rc,
                         &r->d.pgx,&r->d.pgy,&r->d.pgz,&r->d.qgx,&r->d.qgy,&r->d.qgz,&r->d.grad_w_inv,
                         &r->out,&r->grad_buffer}) fields+=v->capacity()*sizeof(scalar_t);
        for(const auto &v:r->state)harness+=v.capacity()*sizeof(scalar_t);
        for(const auto &v:r->direction)harness+=v.capacity()*sizeof(scalar_t);
        // Slots 0/1/2 hold 4 fields each; 3 holds xyz+pg; 4 holds qg;
        // reconstruction slots 7/8 hold one input and three outputs.
        const size_t arena=r->packed?size_t(threads)*25*size_t(r->p.max_actual_nodes_per_pack)*sizeof(scalar_t):0;
        std::printf("payload,layout=%s,field_capacity_bytes=%zu,packed_arena_bytes=%zu,harness_capacity_bytes=%zu\n",
            r->packed?"packed":"atomic",fields,arena,harness);
    }
    const auto pm=packed.p.packed;
    const size_t topology=pm->elements(0)->nbytes()+pm->owned_nodes_ptr(0)->nbytes()+
        pm->n_shared(0)->nbytes()+pm->ghost_ptr(0)->nbytes()+pm->ghost_idx(0)->nbytes()+
        pm->ghost_reduce_ptr(0)->nbytes()+pm->ghost_reduce_idx(0)->nbytes()+
        pm->ghost_reduce_dest(0)->nbytes()+pm->node_map()->nbytes();
    std::printf("storage,atomic_mesh=%zu,packed_retained_mesh=%zu,additional_packed_topology=%zu,ghost_buffer=%zu\n",
        mesh_bytes(atomic),mesh_bytes(packed),topology,packed.p.ghost_buf.size()*sizeof(scalar_t));
    std::printf("setup,atomic_ordering=%.9g,packed_ordering=%.9g,packing=%.9g,packs=%td,ghost_entries=%td,max_pack_nodes=%td,ghost_rows=%td\n",
        atomic.ordering_seconds,packed.ordering_seconds,packed.packing_seconds,packed.p.n_packs,packed.p.n_ghost_entries,packed.p.max_actual_nodes_per_pack,packed.p.n_ghost_reduce_rows);
    if(verify_only) {std::puts("UNSTRUCTURED NUMERICAL CHECKS PASSED");return 0;}
    // Both layouts retain their allocation and alternate order in each block.
    for(int op=0;op<2;++op) {
        for(auto *r:{&atomic,&packed})for(int w=0;w<10;++w)apply(*r,op,w%3);
        for(int block=0;block<blocks;++block)for(int arm=0;arm<2;++arm) {
            bool use_packed=((block+arm)%2)!=0;auto &r=use_packed?packed:atomic;
            for(int w=0;w<warmup;++w)apply(r,op,w%3);
            const auto start=std::chrono::steady_clock::now();
            for(int n=0;n<repeat;++n)apply(r,op,n%3);
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeat;
            require(seconds>0&&std::isfinite(seconds),"invalid elapsed time");
            auto actual=output(r,op);
            if(use_packed) {auto copy=actual;for(ptrdiff_t i=0;i<r.d.nnodes;++i)for(int c=0;c<4;++c)actual[4*i+c]=copy[4*map[i]+c];}
            const double err=relative_error(actual,reference[op][op?(repeat-1)%3:0]);
            require(err<1e-11,"post-timing numerical check failed");
            std::printf("sample,op=%s,block=%d,order=%d,layout=%s,repeat=%d,seconds=%.12g,error=%.12g\n",
                op?"jacobian":"residual",block,arm,use_packed?"packed":"atomic",repeat,seconds,err);
            std::fflush(stdout);
        }
    }
    std::puts("UNSTRUCTURED OPERATOR STUDY PASSED");return 0;
}

int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);int size=0;MPI_Comm_size(MPI_COMM_WORLD,&size);int result=1;
    try {require(size==1,"single MPI rank required");result=run(argc,argv);}
    catch(const std::exception &e) {std::fprintf(stderr,"UNSTRUCTURED CHECK FAILED: %s\n",e.what());}
    MPI_Finalize();return result;
}
