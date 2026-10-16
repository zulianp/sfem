# Mooney-Rivlin gravity test

Runs the elasticity playground (`drivers/mech/playground.exe.cpp`) on `cantilever.yaml` or
`contact.yaml` and visualises it. The material is the Prony-series Mooney-Rivlin, integrated in
time with implicit Newmark (beta = 1/4, gamma = 1/2). The cases were the two modes of
`sfem_MooneyRivlinGravityTest`, which the playground replaced.

```bash
workflows/mooney_rivlin_gravity/run.sh [cantilever|contact] [output-dir] [key=value ...]
```

## Cases

**`cantilever`** (`cantilever.yaml`) is the case the visco history comparisons used:

- **Geometry:** a 10 x 1 x 1 HEX8 box, clamped on the min-x face.
- **Load:** sags under gravity along -y, with density 1 and g = 0.5.
- **Material:** C10 = 800.622, C01 = 800.108, K = 4000; g = 0.4, 0.4, 0.1, 0.05 and
  tau = 1, 2, 5, 10.
- **Time:** dt = 0.005 to T = 7.5.
- **Size:** 640 elements and 3,075 dofs by default, about 20 s on a laptop.

**`contact`** (`contact.yaml`):

- **Geometry:** a free 2 x 1 x 1 block, with no Dirichlet conditions. It falls 0.05 under
  gravity onto a hemisphere of radius 0.25 that stands on a floor. Every node of its
  surface is a contact node.
- **Material and load:** density 10, g = 9.81, C10 = 4000, C01 = 2000, K = 160000. A
  softer block is punched by the hemisphere once the mesh resolves it; with C10 = 50 its
  elements invert at 24 elements per edge.
- **Time:** dt = 0.01 to T = 2.
- **Size:** 512 elements and 2,187 dofs by default, under 30 s.
- **What it shows:** the block lands on the hemisphere and keeps bouncing on it; it is
  stiff and lightly damped. Averaged over time, the obstacle's reaction carries the
  block's weight.

`run.sh --help` lists the options; every `key=value` after the output directory is a
playground `--set`. `obstacle.shape.type` picks a `plane`, a sloping plane (`slope`), a
`staircase` or the `hemisphere`.

On the sloping plane (`obstacle.shape.type=slope`, slope 0.2) the block slides down it. Its
fitted acceleration along x is -1.8880, against the frictionless incline's
-g sin(theta) cos(theta) = -1.8865.

`mesh.box.n` refines the body; with `contact` it is the number of elements along each edge of
the block (8 by default). Measured with the defaults on a laptop, 8 threads, with the retired
test:

| Resolution | Dofs | Max Newton iterations per step | Reaction / weight, mean over t > 1 | Time |
|---|---|---|---|---|
| 8 (default) | 2,187 | 5 | 0.96 | 30 s |
| 12 | 6,591 | 5 | 0.96 | 50 s |
| 16 | 14,739 | 6 | 0.92 | 80 s |
| 24 | 46,875 | 11 | 0.92 | 5 min |

The mean is below 1 because the block spends part of the time in the air.

## Output

The default output is `workflows/mooney_rivlin_gravity/output/<case>`:

```text
run.log                 the playground's log
history.csv             per step: Newton and linear iterations, the final measure; the tip
                        (cantilever) or the obstacle's force and whether it touched (contact)
mesh/, out/             the mesh and the raw fields
output.xdmf, output.h5  disp/velocity/acceleration over time for ParaView;
                        with contact also gap (to the obstacle) and contact_force
sdf/sdf.xdmf            contact: the obstacle's signed distance; contour it at 0
response.csv            cantilever: tip displacement; contact: the mean descent of
                        the underside and the obstacle's reaction over the weight
newton.csv              Newton iterations and the final measure (KKT with contact) per step
gravity_response.png    the response and the Newton iterations over time
deformed.png            the last frame, deformed and coloured by |u| (needs pvbatch)
```

The run stops when a step does not converge. The workflow then stops and shows the
end of the log.

## Contact

The contact is frictionless and node-to-obstacle.

- **Constraint:** every node of the side facing the obstacle keeps
  gap = -sdf(X + u) / |grad sdf| >= 0. The obstacle is a signed distance grid, sampled
  through the library's interpolating cubic B-spline (`resampling/surface/sfem_sdf_sampling.hpp`),
  the one `SDFObstacle` uses. Its gradient and Hessian are continuous, unlike a trilinear
  interpolant's. The grid is the obstacle: a node outside it is out of reach.
- **Step:** each Newton step solves the linearised QP with MPRGP. Each contact node is
  rotated into its frame [n, t1, t2], where the constraint n . dx <= gap is a bound on
  one component.
- **Tangent:** the step includes the obstacle's curvature, lambda * hess(sdf). Without it
  Newton converges only linearly on a curved obstacle.
- **Convergence:** a step converges on its KKT residual: the residual with the obstacle's
  admissible normal reaction removed, plus the complementarity min(mu, c0 m gap) of the
  nodes clear of it, both forces, below 1e-9 of the forces in balance. Penetration must
  also stay below 1e-10 of the mesh size. A step that does not converge stops the run. In the
playground this is `sfem::NodeObstacleStep` (`frontend/mech`), the obstacle's
`LinearizedStep`, and the shapes are `sfem::ObstacleShape`.
- **Globalisation:** a backtracking line search on that residual guards the full step.
  Newton starts from u + dt v, which the stabilised velocity keeps admissible, and falls
  back to u_n if that start cannot be evaluated. When the tangent is indefinite during an
  impact, MPRGP stops; the step is then solved again with the curvature term capped and,
  if needed, the diagonal shifted.
- **Time integration:** contact-stabilised Newmark, after Deuflhard, Krause and Ertel
  (2008) in Klapproth's form (ZIB-Report 12-18, eqs. 12d-12f). The reaction is implicit
  only, so the acceleration carried forward is the material's. The velocity is projected
  so the next predicted position is admissible. Plain Newmark carries the reaction into
  the next predictor and chatters. The material holds the method as a time scheme,
  `sfem::ContactStabilisedNewmarkScheme` (`frontend/ops`), with the contact
  `sfem::NodeObstacleContact` (`frontend`); the cantilever case holds `sfem::NewmarkScheme`.
- **Contact force:** the reported force is half the contact residual. In this
  trapezoidal form the material force enters at both ends of the step and the contact
  force once.
- **Obstacle shape:** a node is held along one normal. Concave creases and convex edges
  of an obstacle are therefore rounded with `obstacle.shape.fillet`; a jump in the normal
  makes Newton cycle.
