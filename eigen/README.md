# WIP for spectral graph processing



Literature

Graph Signal Processing:
Overview, Challenges, and
Applications

DOI: 10.1109/JPROC.2018.2820126

Idea:
	1) Compute poisson problem (on surface?)
		- BC Dirichlet 0 inlet, 1 outlet
	2) Compute velocity field
		- Compute convection operator using grad u (u is solution of Poisson's)
	3) Eigen decomposition of advection operator (Upwind)