#!/usr/bin/env python3
"""
invertible_isochoric_homotopy_2d.py

Structured-mesh inversion recovery test for an uncoupled 2D isochoric
Neo-Hookean energy with a C^2 continuation through J <= 0.

Physical model for J >= Jc:
    W(F) = mu/2 * (I1/J - 2) + kappa/2 * (J - 1)^2.

Thus the physically admissible branch is exactly the standard uncoupled
2D unimodular Neo-Hookean model.

For J < Jc, h(J)=1/J is replaced by a C^2 cubic continuation p(J).
The continuation is constructed on [Jmin, Jc] to satisfy

    p(Jc)   = 1/Jc
    p'(Jc)  = -1/Jc^2
    p''(Jc) =  2/Jc^3
    p'(Jmin)= 0

and is extended linearly-constant in slope for J < Jmin:
    p(J) = p(Jmin)

The derivative flattening on the far-inverted side avoids the very large
artificial distortional forces produced by a pure quadratic Taylor extension.

Solver globalization:
  * absolute-eigenvalue projected Newton
  * Armijo line search with NO inversion barrier
  * continuation in Jc only.
    No positional tether or reference-state penalty is used at any stage.

Only NumPy and Matplotlib are required.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def cubic_continuation_coeffs(Jc, Jmin):
    """
    p(J) = a0 + a1 t + a2 t^2 + a3 t^3, t = J-Jc.
    Match p,p',p'' to 1/J at Jc, and impose p'(Jmin)=0.
    """
    a0 = 1.0 / Jc
    a1 = -1.0 / (Jc * Jc)
    a2 = 1.0 / (Jc**3)  # because p''(Jc)=2*a2=2/Jc^3
    tm = Jmin - Jc
    # p'(tm)=a1+2 a2 tm+3 a3 tm^2 = 0
    a3 = -(a1 + 2.0 * a2 * tm) / (3.0 * tm * tm)
    return a0, a1, a2, a3


def continuation(J, Jc, Jmin):
    if J >= Jc:
        return 1.0/J, -1.0/(J*J), 2.0/(J*J*J)

    a0, a1, a2, a3 = cubic_continuation_coeffs(Jc, Jmin)

    if J <= Jmin:
        tm = Jmin - Jc
        p = a0 + a1*tm + a2*tm*tm + a3*tm**3
        return p, 0.0, 0.0

    t = J - Jc
    p = a0 + a1*t + a2*t*t + a3*t**3
    pp = a1 + 2*a2*t + 3*a3*t*t
    ppp = 2*a2 + 6*a3*t
    return p, pp, ppp


def cofactor_2d(F):
    return np.array([[F[1, 1], -F[1, 0]],
                     [-F[0, 1], F[0, 0]]], dtype=float)


def structured_square_mesh(nx, ny):
    X = np.array([[i/nx, j/ny]
                  for j in range(ny+1)
                  for i in range(nx+1)], dtype=float)

    def node(i, j):
        return j*(nx+1)+i

    tris = []
    for j in range(ny):
        for i in range(nx):
            n00=node(i,j); n10=node(i+1,j); n01=node(i,j+1); n11=node(i+1,j+1)
            if (i+j)&1:
                tris.append([n00,n10,n01])
                tris.append([n10,n11,n01])
            else:
                tris.append([n00,n10,n11])
                tris.append([n00,n11,n01])

    return X, np.asarray(tris, dtype=int), node(nx,0)


def element_kinematics(xe, Xe):
    Dm = np.column_stack((Xe[1]-Xe[0], Xe[2]-Xe[0]))
    inv_Dm = np.linalg.inv(Dm)
    Ds = np.column_stack((xe[1]-xe[0], xe[2]-xe[0]))
    F = Ds @ inv_Dm
    A0 = 0.5*abs(np.linalg.det(Dm))
    return F, inv_Dm, A0


def element_B(inv_Dm):
    B = np.zeros((4,6))
    for j in range(6):
        dxe = np.zeros((3,2))
        dxe.flat[j]=1.0
        dDs = np.column_stack((dxe[1]-dxe[0], dxe[2]-dxe[0]))
        dF = dDs @ inv_Dm
        B[:,j]=dF.ravel()
    return B


def element_energy_gradient_hessian(xe, Xe, mu, kappa, Jc, Jmin):
    F, inv_Dm, A0 = element_kinematics(xe, Xe)
    J = np.linalg.det(F)
    I1 = np.sum(F*F)
    h, hp, hpp = continuation(J, Jc, Jmin)

    W = 0.5*mu*(h*I1 - 2.0) + 0.5*kappa*(J-1.0)**2

    G = cofactor_2d(F)
    q = 0.5*mu*hp*I1 + kappa*(J-1.0)

    P = mu*h*F + q*G

    HF = np.zeros((4,4))
    for j in range(4):
        dF = np.zeros((2,2))
        dF.flat[j]=1.0

        dJ = np.sum(G*dF)
        FdF = np.sum(F*dF)
        dG = cofactor_2d(dF)

        dq = (0.5*mu*hpp*I1 + kappa)*dJ + mu*hp*FdF
        dP = mu*h*dF + mu*hp*dJ*F + dq*G + q*dG
        HF[:,j] = dP.ravel()

    HF = 0.5*(HF+HF.T)
    B = element_B(inv_Dm)
    ge = A0*(B.T @ P.ravel())
    He = A0*(B.T @ HF @ B)
    Ee = A0*W
    return Ee, ge, He, J


def assemble(x, X, triangles, mu, kappa, Jc, Jmin):
    ndof = 2*len(X)
    E=0.0
    g=np.zeros(ndof)
    H=np.zeros((ndof,ndof))
    Js=[]

    for tri in triangles:
        Ee,ge,He,J = element_energy_gradient_hessian(
            x[tri], X[tri], mu, kappa, Jc, Jmin)
        E += Ee
        Js.append(J)
        idx=np.array([[2*i,2*i+1] for i in tri]).ravel()
        g[idx]+=ge

        He = projected_hessian(He)
        H[np.ix_(idx,idx)] += He


    H=0.5*(H+H.T)
    return E,g,H,np.asarray(Js)


def projected_hessian(H, floor=1e-10):
    lam,Q=np.linalg.eigh(H)
    lam_mod=np.maximum(np.abs(lam), floor)
    return (Q*lam_mod)@Q.T


def solve_stage(x, X, triangles, fixed, mu, kappa, Jc, Jmin,
                max_iter=100, grad_tol=1e-9, verbose=True):
    ndof=2*len(X)
    all_dofs=np.arange(ndof)
    free=np.setdiff1d(all_dofs, fixed)

    history=[]

    for it in range(max_iter):
        E,g,H,Js=assemble(x,X,triangles,mu,kappa,Jc,Jmin)
        gf=g[free]
        Hf=H[np.ix_(free,free)]
        gn=np.linalg.norm(gf)
        history.append((E,gn,Js.min(),Js.max()))

        if verbose:
            print(f"{it:3d} E={E: .8e} |g|={gn: .3e} "
                  f"J=[{Js.min(): .5f},{Js.max(): .5f}]")

        if gn < grad_tol:
            return x, history, True

        # Hp=projected_hessian(Hf)
        # p=np.linalg.solve(Hp,-gf)

        p=np.linalg.solve(Hf,-gf)
        gd=float(gf@p)

        if gd >= 0:
            # Fallback to steepest descent.
            p = -gf
            gd = -float(gf@gf)

        flat=x.ravel().copy()
        alpha=1.0
        accepted=False

        for _ in range(80):
            tf=flat.copy()
            tf[free]+=alpha*p
            xt=tf.reshape(x.shape)
            Et,_,_,_=assemble(xt,X,triangles,mu,kappa,Jc,Jmin)
            if np.isfinite(Et) and Et <= E + 1e-4*alpha*gd:
                x=xt
                accepted=True
                break
            alpha*=0.5

        if not accepted:
            return x, history, False

    return x, history, False


def run_homotopy(X, triangles, x0, anchor_right, mu, kappa,
                 Jc_target, Jmin, verbose=True):
    """Tether-free continuation in Jc only."""
    fixed=np.array([0,1,2*anchor_right+1],dtype=int)
    x=x0.copy()

    stages = [0.50, 0.35, 0.25, max(Jc_target,0.20), Jc_target]
    all_hist=[]

    for s,Jc in enumerate(stages):
        if verbose:
            print(f"\n--- stage {s}: Jc={Jc:g}, eta=0 ---")
        x,hist,ok=solve_stage(
            x,X,triangles,fixed,mu,kappa,Jc,Jmin,
            max_iter=160,grad_tol=1e-9,verbose=verbose)
        all_hist.append((Jc,hist))
        if not ok:
            print("Stage failed.")
            return x,all_hist,False

    if verbose:
        print(f"\n--- final exact stage: Jc={Jc_target:g}, eta=0 ---")
    x,hist,ok=solve_stage(
        x,X,triangles,fixed,mu,kappa,Jc_target,Jmin,
        max_iter=200,grad_tol=1e-11,verbose=verbose)
    all_hist.append((Jc_target,hist))
    return x,all_hist,ok

def plot_mesh(ax,x,tris,title):
    for tri in tris:
        pts=np.vstack((x[tri],x[tri[0]]))
        ax.plot(pts[:,0],pts[:,1],"-",linewidth=0.6)
    ax.set_aspect("equal",adjustable="box")
    ax.set_title(title)
    ax.grid(True)



def random_deformed_initial_state(X, triangles, amplitude, seed):
    """
    Reproducible nodal random deformation.

    x0 = X + amplitude * characteristic_length * N(0,1)

    The lower-left node is reset to the origin and the lower-right node's y
    coordinate is reset to zero so the initial state is compatible with the
    rigid-mode constraints.

    Returns x0 plus initial J statistics.
    """
    rng = np.random.default_rng(seed)

    # Characteristic nodal spacing for scale-independent perturbations.
    ux = np.unique(X[:, 0])
    uy = np.unique(X[:, 1])
    hx = np.min(np.diff(ux)) if len(ux) > 1 else 1.0
    hy = np.min(np.diff(uy)) if len(uy) > 1 else 1.0
    h = min(hx, hy)

    x0 = X + amplitude * h * rng.standard_normal(X.shape)

    # Match the constraints used by the nonlinear solve.
    lower_left = np.argmin(np.sum(X * X, axis=1))
    lower_right = np.argmax(X[:, 0] - 1000.0 * np.abs(X[:, 1]))
    x0[lower_left] = X[lower_left]
    x0[lower_right, 1] = X[lower_right, 1]

    Js = []
    for tri in triangles:
        F, _, _ = element_kinematics(x0[tri], X[tri])
        Js.append(np.linalg.det(F))
    Js = np.asarray(Js)

    return x0, Js


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nx",type=int,default=8)
    ap.add_argument("--ny",type=int,default=8)
    ap.add_argument("--mu",type=float,default=1.0)
    ap.add_argument("--kappa",type=float,default=20.0)
    ap.add_argument("--Jc",type=float,default=0.2)
    ap.add_argument("--Jmin",type=float,default=-1.0)
    ap.add_argument("--plot",default="invertible_isochoric_random.png")
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--amplitude",type=float,default=1.5,
                    help="Random nodal displacement amplitude in units of mesh spacing.")
    args=ap.parse_args()

    X,tris,anchor_right=structured_square_mesh(args.nx,args.ny)
    x0, Js0 = random_deformed_initial_state(
        X, tris, amplitude=args.amplitude, seed=args.seed
    )

    ninv = int(np.count_nonzero(Js0 < 0.0))
    print(f"Initial random deformation: seed={args.seed}, amplitude={args.amplitude}")
    print(f"Initial J range: [{Js0.min():.6f}, {Js0.max():.6f}]")
    print(f"Initially inverted elements: {ninv}/{len(Js0)}")

    x,hist,ok=run_homotopy(
        X,tris,x0,anchor_right,args.mu,args.kappa,args.Jc,args.Jmin,True)

    E,g,H,Js=assemble(x,X,tris,args.mu,args.kappa,args.Jc,args.Jmin)
    fixed=np.array([0,1,2*anchor_right+1],dtype=int)
    free=np.setdiff1d(np.arange(2*len(X)),fixed)

    print("\nConverged:",ok)
    print(f"Mesh: {len(X)} vertices, {len(tris)} triangles")
    print("Final J range:",(Js.min(),Js.max()))
    print("Final energy:",E)
    print("Final free-gradient norm:",np.linalg.norm(g[free]))
    print("||x-X||_F:",np.linalg.norm(x-X))

    fig,axs=plt.subplots(1,3,figsize=(12,4))
    plot_mesh(axs[0],x0,tris,"initial random deformation")
    plot_mesh(axs[1],x,tris,"final")

    stage_ids=[]; energies=[]; jmins=[]; jmaxs=[]
    k=0
    for Jc,hh in hist:
        for E0,gn,jmn,jmx in hh:
            stage_ids.append(k); energies.append(max(E0,1e-16))
            jmins.append(jmn); jmaxs.append(jmx); k+=1

    axs[2].semilogy(stage_ids,energies,label="energy")
    ax2=axs[2].twinx()
    ax2.plot(stage_ids,jmins,"--",label="min J")
    ax2.plot(stage_ids,jmaxs,":",label="max J")
    axs[2].set_xlabel("global iteration")
    axs[2].set_ylabel("energy")
    ax2.set_ylabel("J")
    axs[2].grid(True)

    l1,lab1=axs[2].get_legend_handles_labels()
    l2,lab2=ax2.get_legend_handles_labels()
    axs[2].legend(l1+l2,lab1+lab2,loc="best")
    fig.tight_layout()
    fig.savefig(args.plot,dpi=180)
    print("Saved plot:",args.plot)

    if not ok:
        raise SystemExit(1)


if __name__=="__main__":
    main()
