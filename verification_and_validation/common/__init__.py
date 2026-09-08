"""Shared infrastructure for SFEM verification and validation cases."""

from .mesh import Mesh, read_mesh, write_mesh

__all__ = ["Mesh", "read_mesh", "write_mesh"]
