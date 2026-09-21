"""Records describing what a generator produced.

A ``GeneratedKernelCode`` is source text still in memory; a
``GeneratedKernelFile`` is that text bound to a filename.  Both are outputs of
the emission layer, so neither belongs in the specification layer where they
used to live.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedKernelCode:
    language: str
    function_name: str
    source: str


@dataclass(frozen=True)
class GeneratedKernelFile:
    path: str
    source: str
