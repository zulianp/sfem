#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
using scalar_t = double;
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"

// Independent mapped-point oracle, scalar/SIMD agreement and directional differences.
int main() {
    using T = double;
    constexpr int W = cvfem_hex8_vec_size<T>;
    Hex8CoordPackT<T> xyz;
    Hex8InputPackT<T> u, v;
    Hex8UGradPackT<T> ug, vg;
    Hex8ResidualPackT<T> residual, action;
    std::array<std::array<T,32>,W> scalar_r, scalar_j;
    double worst_fd=0, worst_layout=0, correction=0;
    // The physical construction must reduce to the existing affine increments.
    {
        T x[8],y[8],z[8],g[72],reference[72];
        const T ex[3]={1.2,.1,.2},ey[3]={.2,.9,.1},ez[3]={.1,.2,1.1};
        for(int a=0;a<8;++a) {
            x[a]=y[a]=z[a]=0;
            for(int k=0;k<3;++k) {x[a]+=ex[k]*CVFEM_HEX8_REF_XI[a][k];y[a]+=ey[k]*CVFEM_HEX8_REF_XI[a][k];z[a]+=ez[k]*CVFEM_HEX8_REF_XI[a][k];}
            for(int c=0;c<9;++c) g[a*9+c]=reference[a*9+c]=.1+.02*a-.03*c;
        }
        cvfem_hex8_defcor_ref_increments<1>(reference,ex,ey,ez);
        cvfem_hex8_defcor_isoparam_increments<1>(g,x,y,z);
        for(int k=0;k<72;++k) if(std::abs(g[k]-reference[k])>1e-14)
            throw std::runtime_error("affine reconstruction changed");
    }

    for(int lane=0;lane<W;++lane) {
        T x[8],y[8],z[8],ux[8],uy[8],uz[8],p[8],vx[8],vy[8],vz[8],q[8],g[72],dg[72];
        for(int a=0;a<8;++a) {
            x[a]=CVFEM_HEX8_REF_XI[a][0]+(a==6 ? .2+.003*lane : 0);
            y[a]=CVFEM_HEX8_REF_XI[a][1]+(a==5 ? .08 : 0);
            z[a]=CVFEM_HEX8_REF_XI[a][2]+(a==7 ? .11 : 0);
            xyz.x[a][lane]=x[a];xyz.y[a][lane]=y[a];xyz.z[a][lane]=z[a];
            ux[a]=u.ux[a][lane]=.8+.13*a+.007*lane;
            uy[a]=u.uy[a][lane]=.4+.03*a*a;
            uz[a]=u.uz[a][lane]=.9-.04*a;
            p[a]=u.p[a][lane]=.1+.05*a;
            vx[a]=v.ux[a][lane]=.03*std::sin(a+.3);
            vy[a]=v.uy[a][lane]=.05*std::cos(a+.2);
            vz[a]=v.uz[a][lane]=.02+.01*a;
            q[a]=v.p[a][lane]=.04*std::sin(a+.7);
            for(int c=0;c<9;++c) {
                g[a*9+c]=ug.g[a][c][lane]=.21+.013*a+.019*c;
                dg[a*9+c]=vg.g[a][c][lane]=.011*std::cos(a+c+.2);
            }
        }
        T increments[72];std::copy(g,g+72,increments);
        cvfem_hex8_defcor_isoparam_increments<1>(increments,x,y,z);
        for(int s=0;s<12;++s) {
            T face[3]={};
            for(int a=0;a<8;++a) {
                T weight=1;
                for(int k=0;k<3;++k) weight*=CVFEM_HEX8_REF_XI[a][k] ? CVFEM_HEX8_SCS_XI[s][k] : 1-CVFEM_HEX8_SCS_XI[s][k];
                face[0]+=weight*x[a];face[1]+=weight*y[a];face[2]+=weight*z[a];
            }
            for(int a:{CVFEM_HEX8_SCS[s].i,CVFEM_HEX8_SCS[s].j})for(int c=0;c<3;++c) {
                T expected=g[a*9+c*3]*(face[0]-x[a])+g[a*9+c*3+1]*(face[1]-y[a])+g[a*9+c*3+2]*(face[2]-z[a]);
                if(std::abs(expected-increments[a*9+c*3+(s>>2)])>1e-14)
                    throw std::runtime_error("wrong physical reconstruction point");
            }
        }
        T gc[72],dgc[72],bare[32];std::copy(g,g+72,gc);std::copy(dg,dg+72,dgc);
        cvfem_hex8_ns_upwind_residual_isoparam(T(1.3),T(.027),x,y,z,ux,uy,uz,p,scalar_r[lane].data(),{},T(0),gc);
        cvfem_hex8_ns_upwind_residual_isoparam(T(1.3),T(.027),x,y,z,ux,uy,uz,p,bare);
        for(int k=0;k<32;++k)correction=std::max(correction,std::abs(bare[k]-scalar_r[lane][k]));
        std::copy(g,g+72,gc);
        cvfem_hex8_ns_upwind_jacobian_action_isoparam(T(1.3),T(.027),x,y,z,ux,uy,uz,vx,vy,vz,q,scalar_j[lane].data(),{},p,T(0),gc,dgc);
        for(T eps:{1e-5,1e-6}) {
            T plus[32],minus[32];
            for(int sign:{-1,1}) {
                T ax[8],ay[8],az[8],ap[8],ag[72];
                for(int a=0;a<8;++a) {ax[a]=ux[a]+sign*eps*vx[a];ay[a]=uy[a]+sign*eps*vy[a];az[a]=uz[a]+sign*eps*vz[a];ap[a]=p[a]+sign*eps*q[a];}
                for(int k=0;k<72;++k)ag[k]=g[k]+sign*eps*dg[k];
                cvfem_hex8_ns_upwind_residual_isoparam(T(1.3),T(.027),x,y,z,ax,ay,az,ap,sign>0?plus:minus,{},T(0),ag);
            }
            for(int k=0;k<32;++k) {
                if(!std::isfinite(plus[k]) || !std::isfinite(minus[k]) || !std::isfinite(scalar_j[lane][k]))
                    throw std::runtime_error("nonfinite derivative check");
                worst_fd=std::max(worst_fd,std::abs((plus[k]-minus[k])/(2*eps)-scalar_j[lane][k]));
            }
        }
    }
    auto rg=ug,jg=ug,jv=vg;
    cvfem_hex8_ns_upwind_residual_isoparam_simd(T(1.3),T(.027),xyz,u,residual,T(0),static_cast<const Hex8RhieChowPackT<T>*>(nullptr),T(0),&rg);
    cvfem_hex8_ns_upwind_jacobian_action_isoparam_simd(T(1.3),T(.027),xyz,u,v,action,T(0),static_cast<const Hex8RhieChowPackT<T>*>(nullptr),T(0),false,&jg,&jv);
    for(int a=0;a<8;++a)for(int lane=0;lane<W;++lane) {
        const T r[4]={residual.rx[a][lane],residual.ry[a][lane],residual.rz[a][lane],residual.rc[a][lane]};
        const T j[4]={action.rx[a][lane],action.ry[a][lane],action.rz[a][lane],action.rc[a][lane]};
        for(int c=0;c<4;++c) {
            if(!std::isfinite(r[c]) || !std::isfinite(j[c]) || !std::isfinite(scalar_r[lane][a*4+c]))
                throw std::runtime_error("nonfinite layout check");
            worst_layout=std::max({worst_layout,std::abs(r[c]-scalar_r[lane][a*4+c]),std::abs(j[c]-scalar_j[lane][a*4+c])});
        }
    }
    std::printf("isoparametric Darwish: FD=%.12g scalar/SIMD=%.12g correction=%.12g\n",worst_fd,worst_layout,correction);
    return !(std::isfinite(worst_fd)&&worst_fd<1e-8&&worst_layout<1e-12&&correction>1e-3);
}
