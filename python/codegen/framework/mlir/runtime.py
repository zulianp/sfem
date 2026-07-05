from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import threading

import numpy as np

from .model import pad_to_vector_width


@dataclass(frozen=True)
class InvertedTopology:
    node_degree: np.ndarray
    node_to_element_map: np.ndarray
    node_to_local_idx: np.ndarray


@dataclass(frozen=True)
class EBEExecutionResult:
    residual: np.ndarray
    scratchpad: np.ndarray
    map_worker_count: int
    reduce_worker_count: int
    padded_scratch_components: int

class ThreadedEBEExecutor:
    def __init__(self, model, max_workers=None, synchronize_workers=False):
        self.model = model
        self.max_workers = max_workers
        self.synchronize_workers = bool(synchronize_workers)

    def run(self, current, direction, connectivity, local_apply, parameters=None, inverted=None):
        current = np.asarray(current, dtype=np.float32)
        direction = np.asarray(direction, dtype=np.float32)
        connectivity = np.asarray(connectivity, dtype=np.intp)
        if connectivity.ndim != 2:
            raise ValueError("connectivity must be rank-2")
        if current.ndim != 2 or direction.ndim != 2:
            raise ValueError("current and direction must be rank-2 node-component arrays")
        if current.shape != direction.shape:
            raise ValueError("current and direction must have identical shapes")
        if current.shape[1] != self.model.n_field_components:
            raise ValueError("field component count does not match the MLIR kernel model")
        if connectivity.shape[1] != self.model.n_shape:
            raise ValueError("connectivity shape count does not match the MLIR kernel model")
        if local_apply is None:
            raise ValueError("local_apply callback is required; the framework local kernel supplies this boundary")

        num_elements = connectivity.shape[0]
        num_nodes = int(current.shape[0])
        if inverted is None:
            inverted = build_inverted_topology(connectivity, num_nodes)

        padded = self.model.padded_scratch_components
        scratchpad = np.zeros((num_elements, self.model.n_shape, padded // self.model.n_shape), dtype=np.float32)
        residual = np.zeros((num_nodes, self.model.n_field_components), dtype=np.float32)
        params = {} if parameters is None else dict(parameters)

        map_workers = self._parallel_element_map(
            current,
            direction,
            connectivity,
            scratchpad,
            local_apply,
            params,
        )
        reduce_workers = self._parallel_node_reduce(inverted, scratchpad, residual)

        return EBEExecutionResult(
            residual=residual,
            scratchpad=scratchpad,
            map_worker_count=map_workers,
            reduce_worker_count=reduce_workers,
            padded_scratch_components=padded,
        )

    def _parallel_element_map(self, current, direction, connectivity, scratchpad, local_apply, parameters):
        num_elements = connectivity.shape[0]
        chunks = _chunks(num_elements, self._worker_count(num_elements))
        barrier = _barrier_for_chunks(chunks, self.synchronize_workers)
        thread_ids = set()
        lock = threading.Lock()

        def work(begin_end):
            _wait_at_barrier(barrier)
            begin, end = begin_end
            with lock:
                thread_ids.add(threading.get_ident())
            for elem in range(begin, end):
                for local_node in range(self.model.n_shape):
                    node = int(connectivity[elem, local_node])
                    for component in range(self.model.n_field_components):
                        scratchpad[elem, local_node, component] = np.float32(
                            local_apply(
                                elem,
                                local_node,
                                component,
                                node,
                                current,
                                direction,
                                parameters,
                            )
                        )

        self._run_chunks(chunks, work)
        return len(thread_ids)

    def _parallel_node_reduce(self, inverted, scratchpad, residual):
        num_nodes = residual.shape[0]
        chunks = _chunks(num_nodes, self._worker_count(num_nodes))
        barrier = _barrier_for_chunks(chunks, self.synchronize_workers)
        thread_ids = set()
        lock = threading.Lock()

        def work(begin_end):
            _wait_at_barrier(barrier)
            begin, end = begin_end
            with lock:
                thread_ids.add(threading.get_ident())
            for node in range(begin, end):
                degree = int(inverted.node_degree[node])
                for component in range(self.model.n_field_components):
                    acc = np.float32(0.0)
                    for i in range(degree):
                        elem = int(inverted.node_to_element_map[node, i])
                        local = int(inverted.node_to_local_idx[node, i])
                        acc = np.float32(acc + scratchpad[elem, local, component])
                    residual[node, component] = acc

        self._run_chunks(chunks, work)
        return len(thread_ids)

    def _worker_count(self, work_items):
        if work_items <= 0:
            return 1
        requested = self.max_workers
        if requested is None:
            requested = min(32, (work_items + 1) // 2)
        return max(1, min(int(requested), int(work_items)))

    def _run_chunks(self, chunks, work):
        if not chunks:
            return
        if len(chunks) == 1:
            work(chunks[0])
            return
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(work, chunk) for chunk in chunks]
            for future in futures:
                future.result()


def build_inverted_topology(connectivity, num_nodes=None):
    connectivity = np.asarray(connectivity, dtype=np.intp)
    if connectivity.ndim != 2:
        raise ValueError("connectivity must be rank-2")
    if num_nodes is None:
        num_nodes = int(np.max(connectivity)) + 1 if connectivity.size else 0
    num_nodes = int(num_nodes)

    node_degree = np.zeros(num_nodes, dtype=np.intp)
    for elem in range(connectivity.shape[0]):
        for node in connectivity[elem]:
            node_degree[int(node)] += 1

    max_degree = int(np.max(node_degree)) if num_nodes else 0
    node_to_element_map = np.zeros((num_nodes, max_degree), dtype=np.intp)
    node_to_local_idx = np.zeros((num_nodes, max_degree), dtype=np.intp)
    offsets = np.zeros(num_nodes, dtype=np.intp)

    for elem in range(connectivity.shape[0]):
        for local, node_value in enumerate(connectivity[elem]):
            node = int(node_value)
            offset = int(offsets[node])
            node_to_element_map[node, offset] = elem
            node_to_local_idx[node, offset] = local
            offsets[node] += 1

    return InvertedTopology(node_degree, node_to_element_map, node_to_local_idx)


def pad_to_vector_width(value, vector_width):
    value = int(value)
    vector_width = int(vector_width)
    return int(math.ceil(value / vector_width) * vector_width) if value else 0


def reference_ebe_residual(model, current, direction, connectivity, local_apply, parameters=None):
    current = np.asarray(current, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    connectivity = np.asarray(connectivity, dtype=np.intp)
    residual = np.zeros((current.shape[0], model.n_field_components), dtype=np.float32)
    params = {} if parameters is None else dict(parameters)
    for elem in range(connectivity.shape[0]):
        for local_node in range(model.n_shape):
            node = int(connectivity[elem, local_node])
            for component in range(model.n_field_components):
                residual[node, component] = np.float32(
                    residual[node, component]
                    + local_apply(
                        elem,
                        local_node,
                        component,
                        node,
                        current,
                        direction,
                        params,
                    )
                )
    return residual

def _chunks(count, requested_chunks):
    count = int(count)
    requested_chunks = max(1, int(requested_chunks))
    if count <= 0:
        return []
    chunk_count = min(count, requested_chunks)
    chunk_size = int(math.ceil(count / chunk_count))
    return [
        (begin, min(count, begin + chunk_size))
        for begin in range(0, count, chunk_size)
    ]


def _barrier_for_chunks(chunks, enabled):
    if enabled and len(chunks) > 1:
        return threading.Barrier(len(chunks))
    return None


def _wait_at_barrier(barrier):
    if barrier is not None:
        barrier.wait()
