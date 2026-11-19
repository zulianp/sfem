<!-- RIGID_BODY_MODES.md -->
# Rigid body modes

For every node `p_i = [x_i, y_i, z_i]` in the mesh we can extract the rigid body modes associated with

## Elasticity

```c
M[3*6] = {
	1, 0, 0, 
	0, 1, 0,
	0, 0, 1,
};
```