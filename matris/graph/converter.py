"""
    This code is referenced from: https://github.com/CederGroupHub/chgnet/blob/main/chgnet/graph/converter.py
    The original implementation can be found at the link above.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch import nn
import gc

from .radiusgraph import Graph, Node, RadiusGraph
from pymatgen.core import Structure

try:
    from .cygraph import (
        build_graph_tensors_fast,
        line_graph_adjacency_list_fast,
        make_graph,
    )
except (ImportError, AttributeError):
    build_graph_tensors_fast = None
    make_graph = None
    line_graph_adjacency_list_fast = None

datatype = torch.float32

class GraphConverter(nn.Module):
    """Convert a pymatgen.core.Structure to a RadiusGraph"""

    def __init__(
        self,
        atom_graph_cutoff: float = 6,
        line_graph_cutoff: float = 4,
        neighbor_backend: str | None = None,
        neighbor_device: str | None = None,
        neighbor_method: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the Graph Converter.
        
        Args:
            atom_graph_cutoff (float): cutoff radius in atom graph.
            line_graph_cutoff (float): bond length threshold in line graph.
            neighbor_backend (str): neighbor list backend. Defaults to
                MATRIS_GRAPH_BACKEND or "pymatgen".
            neighbor_device (str): device for nvalchemi neighbor list. Defaults to
                MATRIS_NV_GRAPH_DEVICE or cuda when available.
            neighbor_method (str): nvalchemi neighbor list method. Defaults to
                MATRIS_NV_GRAPH_METHOD or "naive".
            verbose (bool): whether to print the GraphConverter.
        """
        super().__init__()
        self.atom_graph_cutoff = atom_graph_cutoff
        self.line_graph_cutoff = (
            atom_graph_cutoff if line_graph_cutoff is None else line_graph_cutoff
        )
        self.neighbor_backend = (
            neighbor_backend or os.environ.get("MATRIS_GRAPH_BACKEND", "pymatgen")
        ).strip().lower()
        self.neighbor_device = neighbor_device or os.environ.get("MATRIS_NV_GRAPH_DEVICE")
        method = neighbor_method
        if method is None:
            method = os.environ.get("MATRIS_NV_GRAPH_METHOD", "auto")
        method = method.strip().lower() if isinstance(method, str) else method
        self.neighbor_method = None if method in ("", "auto", "none") else method
        
        if make_graph is not None:
            self.create_graph = self._create_graph_fast
            self.algorithm = 'fast'
        else: 
            self.create_graph = self._create_graph_legacy
            self.algorithm = 'legacy'
            print("fast graph converter algorithm import error, using legacy")
        
        if verbose:
            print(self)

    def __repr__(self) -> str:
        """String representation of the GraphConverter."""
        atom_graph_cutoff = self.atom_graph_cutoff
        line_graph_cutoff = self.line_graph_cutoff
        algorithm = self.algorithm
        neighbor_backend = self.neighbor_backend
        neighbor_method = self.neighbor_method
        cls_name = type(self).__name__
        return (
            f"{cls_name}({algorithm=}, {neighbor_backend=}, {neighbor_method=}, "
            f"{atom_graph_cutoff=}, {line_graph_cutoff=})"
        )

    def forward(
        self,
        structure: Structure,
        graph_id=None,
        mp_id=None,
    ) -> RadiusGraph:
        """Convert a structure, return a RadiusGraph.

        Args:
            structure (pymatgen.core.Structure): structure to convert
            graph_id (str): an id to keep track of this crystal graph
                Default = None
            mp_id (str): Materials Project id of this structure
                Default = None
        
        """
        n_atoms = len(structure)
        atomic_number = torch.tensor( [site.specie.Z for site in structure], dtype=torch.int32 )
        
        atom_frac_coord = torch.tensor( structure.frac_coords, dtype=datatype )
        lattice = torch.tensor( structure.lattice.matrix, dtype=datatype )
        
        center_index, neighbor_index, image, distance = self._get_neighbor_list(structure)
        if (
            self.neighbor_backend in ("nvalchemi", "nv", "nvidia")
            and os.environ.get("MATRIS_NV_GRAPH_TENSOR_PATH", "0") == "1"
        ):
            if build_graph_tensors_fast is not None:
                atom_graph, directed2undirected, undirected2directed, line_graph = (
                    self._build_graph_tensors_fast(
                        n_atoms,
                        center_index,
                        neighbor_index,
                        image,
                        distance,
                    )
                )
            else:
                atom_graph, directed2undirected, undirected2directed, line_graph = (
                    self._build_graph_tensors_from_edges(
                        n_atoms,
                        center_index,
                        neighbor_index,
                        image,
                        distance,
                    )
                )
            n_isolated_atoms = len({*range(n_atoms)} - {*center_index})
            if n_isolated_atoms:
                error = f"Error: Detected {n_isolated_atoms} isolated atom. Calculation stopped"
                raise ValueError(error)
            return RadiusGraph(
                atomic_number=atomic_number,
                atom_frac_coord=atom_frac_coord,
                atom_graph=atom_graph,
                neighbor_image=torch.tensor(image, dtype=datatype),
                directed2undirected=directed2undirected,
                undirected2directed=undirected2directed,
                line_graph=line_graph,
                lattice=lattice,
                graph_id=graph_id,
                mp_id=mp_id,
                composition=structure.composition.formula,
                atom_graph_cutoff=self.atom_graph_cutoff,
                line_graph_cutoff=self.line_graph_cutoff,
            )

        # Ceate atom graph
        graph = self.create_graph(
            n_atoms, center_index, neighbor_index, image, distance
        )
        atom_graph, directed2undirected = graph.adjacency_list()
        atom_graph = torch.tensor(atom_graph, dtype=torch.int32)
        directed2undirected = torch.tensor(directed2undirected, dtype=torch.int32)
        undirected2directed = graph.undirected2directed()
        undirected2directed = torch.tensor(undirected2directed, dtype=torch.int32)
        
        line_graph = []
        disable_gc_for_line_graph = os.environ.get(
            "MATRIS_DISABLE_GC_DURING_LINE_GRAPH", "1"
        ) != "0"
        use_fast_line_graph = (
            line_graph_adjacency_list_fast is not None
            and os.environ.get("MATRIS_USE_FAST_LINE_GRAPH", "0") == "1"
        )
        gc_was_enabled = gc.isenabled()
        if disable_gc_for_line_graph and gc_was_enabled:
            gc.disable()
        try:
            try:
                if use_fast_line_graph:
                    line_graph = line_graph_adjacency_list_fast(
                        graph.nodes,
                        graph.undirected_edges_list,
                        self.line_graph_cutoff,
                    )
                else:
                    line_graph = graph.line_graph_adjacency_list(
                        cutoff=self.line_graph_cutoff
                    )
            except Exception as exc:
                structure.to(filename="error_graph.cif")

            line_graph = torch.tensor(line_graph, dtype=torch.int32)
        finally:
            if disable_gc_for_line_graph and gc_was_enabled:
                gc.enable()

        # For isolated atom, we stop this calculation
        n_isolated_atoms = len({*range(n_atoms)} - {*center_index})
        if n_isolated_atoms:
            atom_graph_cutoff = self.atom_graph_cutoff
            error = f"Error: Detected {n_isolated_atoms} isolated atom. Calculation stopped"
            raise ValueError(error) # or print(error)
        
        return RadiusGraph(
            atomic_number=atomic_number,
            atom_frac_coord=atom_frac_coord,
            atom_graph=atom_graph,
            neighbor_image=torch.tensor(image, dtype=datatype),
            directed2undirected=directed2undirected,
            undirected2directed=undirected2directed,
            line_graph=line_graph,
            lattice=lattice,
            graph_id=graph_id,
            mp_id=mp_id,
            composition=structure.composition.formula,
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph_cutoff=self.line_graph_cutoff,
        )

    def _get_neighbor_list(self, structure: Structure):
        if self.neighbor_backend in ("pymatgen", "pmg"):
            return structure.get_neighbor_list(
                r=self.atom_graph_cutoff, sites=structure.sites, numerical_tol=1e-8
            )
        if self.neighbor_backend in ("nvalchemi", "nv", "nvidia"):
            return self._get_neighbor_list_nvalchemi(structure)
        raise ValueError(
            f"Unknown neighbor_backend={self.neighbor_backend!r}. "
            "Expected 'pymatgen' or 'nvalchemi'."
        )

    def _get_neighbor_list_nvalchemi(self, structure: Structure):
        try:
            from nvalchemiops.torch.neighbors import neighbor_list
        except ImportError as exc:
            raise ImportError(
                "MATRIS_GRAPH_BACKEND=nvalchemi requires nvalchemi-toolkit-ops "
                "and warp-lang to be installed."
            ) from exc

        device = self.neighbor_device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        positions = torch.tensor(
            structure.cart_coords,
            dtype=datatype,
            device=device,
        )
        lattice = torch.tensor(
            structure.lattice.matrix,
            dtype=datatype,
            device=device,
        )
        pbc = torch.ones((1, 3), dtype=torch.bool, device=device)

        edge_index, _, shifts = neighbor_list(
            positions,
            self.atom_graph_cutoff,
            cell=lattice.unsqueeze(0),
            pbc=pbc,
            return_neighbor_list=True,
            method=self.neighbor_method,
        )
        if positions.is_cuda:
            torch.cuda.synchronize(positions.device)

        center_index = edge_index[0].detach().cpu().numpy().astype(np.int_, copy=False)
        neighbor_index = edge_index[1].detach().cpu().numpy().astype(np.int_, copy=False)
        image = shifts.detach().cpu().numpy().astype(np.int_, copy=False)

        shift_vectors = shifts.to(dtype=datatype) @ lattice
        edge_vectors = positions[edge_index[0].long()] - (
            positions[edge_index[1].long()] + shift_vectors
        )
        distance = torch.linalg.norm(edge_vectors, dim=1)
        distance = distance.detach().cpu().numpy().astype(np.float64, copy=False)
        return center_index, neighbor_index, image, distance

    def _build_graph_tensors_fast(
        self,
        n_atoms: int,
        center_index: np.ndarray,
        neighbor_index: np.ndarray,
        image: np.ndarray,
        distance: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        center_index = np.ascontiguousarray(center_index, dtype=np.int_)
        neighbor_index = np.ascontiguousarray(neighbor_index, dtype=np.int_)
        image = np.ascontiguousarray(image, dtype=np.int_)
        distance = np.ascontiguousarray(distance, dtype=np.float64)
        (
            atom_graph_np,
            directed2undirected_np,
            undirected2directed_np,
            line_graph_np,
        ) = build_graph_tensors_fast(
            center_index,
            len(center_index),
            neighbor_index,
            image,
            distance,
            n_atoms,
            self.line_graph_cutoff,
        )
        return (
            torch.from_numpy(atom_graph_np),
            torch.from_numpy(directed2undirected_np),
            torch.from_numpy(undirected2directed_np),
            torch.from_numpy(line_graph_np),
        )

    def _build_graph_tensors_from_edges(
        self,
        n_atoms: int,
        center_index: np.ndarray,
        neighbor_index: np.ndarray,
        image: np.ndarray,
        distance: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        center_index = np.asarray(center_index, dtype=np.int64)
        neighbor_index = np.asarray(neighbor_index, dtype=np.int64)
        image = np.asarray(image, dtype=np.int64)
        distance = np.asarray(distance, dtype=np.float64)
        num_edges = int(center_index.shape[0])

        atom_graph_np = np.empty((num_edges, 2), dtype=np.int32)
        atom_graph_np[:, 0] = center_index.astype(np.int32, copy=False)
        atom_graph_np[:, 1] = neighbor_index.astype(np.int32, copy=False)

        directed2undirected_np = np.empty(num_edges, dtype=np.int32)
        undirected2directed = []
        ude_directed_edges: list[list[int]] = []
        ude_nodes: list[tuple[int, int]] = []
        ude_distance: list[float] = []
        key_to_ude: dict[tuple, int] = {}

        def canonical_key(i: int, j: int, img_tuple: tuple[int, int, int]) -> tuple:
            forward = (i, j, img_tuple)
            reverse = (j, i, (-img_tuple[0], -img_tuple[1], -img_tuple[2]))
            return forward if forward <= reverse else reverse

        outgoing: list[list[int]] = [[] for _ in range(n_atoms)]
        for de_idx in range(num_edges):
            i = int(center_index[de_idx])
            j = int(neighbor_index[de_idx])
            img_tuple = (
                int(image[de_idx, 0]),
                int(image[de_idx, 1]),
                int(image[de_idx, 2]),
            )
            key = canonical_key(i, j, img_tuple)
            ude_idx = key_to_ude.get(key)
            if ude_idx is None:
                ude_idx = len(ude_directed_edges)
                key_to_ude[key] = ude_idx
                ude_directed_edges.append([])
                ude_nodes.append((i, j))
                ude_distance.append(float(distance[de_idx]))
                undirected2directed.append(de_idx)
            directed2undirected_np[de_idx] = ude_idx
            ude_directed_edges[ude_idx].append(de_idx)
            outgoing[i].append(de_idx)

        bad_pairs = [idx for idx, edges in enumerate(ude_directed_edges) if len(edges) != 2]
        if bad_pairs:
            raise ValueError(
                "Tensor graph path expected exactly 2 directed edges per undirected edge; "
                f"found {len(bad_pairs)} invalid groups, first={bad_pairs[:5]}"
            )

        line_rows: list[list[int]] = []
        cutoff = float(self.line_graph_cutoff)
        for ude_idx, de_pair in enumerate(ude_directed_edges):
            if ude_distance[ude_idx] > cutoff:
                continue
            for de_idx in de_pair:
                center = int(center_index[de_idx])
                for other_de in outgoing[center]:
                    if other_de == de_idx:
                        continue
                    if float(distance[other_de]) < cutoff:
                        line_rows.append(
                            [
                                center,
                                ude_idx,
                                de_idx,
                                int(directed2undirected_np[other_de]),
                                other_de,
                            ]
                        )

        line_graph_np = np.asarray(line_rows, dtype=np.int32)
        if line_graph_np.size == 0:
            line_graph_np = np.empty((0, 5), dtype=np.int32)

        return (
            torch.from_numpy(atom_graph_np),
            torch.from_numpy(directed2undirected_np),
            torch.tensor(undirected2directed, dtype=torch.int32),
            torch.from_numpy(line_graph_np),
        )

    @staticmethod
    def _create_graph_legacy(
        n_atoms: int,
        center_index: np.ndarray,
        neighbor_index: np.ndarray,
        image: np.ndarray,
        distance: np.ndarray,
    ) -> Graph:
        """Given structure information, create a Graph structure to be used to
        create Crystal_Graph using pure python implementation.

        Args:
            n_atoms (int): the number of atoms in the structure
            center_index (np.ndarray): np array of indices of center atoms.
                [num_undirected_bonds]
            neighbor_index (np.ndarray): np array of indices of neighbor atoms.
                [num_undirected_bonds]
            image (np.ndarray): np array of images for each edge.
                [num_undirected_bonds, 3]
            distance (np.ndarray): np array of distances.
                [num_undirected_bonds]

        Return:
            Graph data structure used to create Crystal_Graph object
        """
        
        graph = Graph([Node(index=idx) for idx in range(n_atoms)])
        for ii, jj, img, dist in zip(center_index, neighbor_index, image, distance):
            graph.add_edge(center_index=ii, neighbor_index=jj, image=img, distance=dist)
      
        return graph

    @staticmethod
    def _create_graph_fast(
        n_atoms: int,
        center_index: np.ndarray,
        neighbor_index: np.ndarray,
        image: np.ndarray,
        distance: np.ndarray,
    ) -> Graph:
        """Given structure information, create a Graph structure to be used to
        create Crystal_Graph using C implementation.

        NOTE: this is the fast version of _create_graph_legacy optimized
            in c (~3x speedup).

        Args:
            n_atoms (int): the number of atoms in the structure
            center_index (np.ndarray): np array of indices of center atoms.
                [num_undirected_bonds]
            neighbor_index (np.ndarray): np array of indices of neighbor atoms.
                [num_undirected_bonds]
            image (np.ndarray): np array of images for each edge.
                [num_undirected_bonds, 3]
            distance (np.ndarray): np array of distances.
                [num_undirected_bonds]
        
        Return:
            Graph data structure used to create Crystal_Graph object
        """
        center_index = np.ascontiguousarray(center_index)
        neighbor_index = np.ascontiguousarray(neighbor_index)
        image = np.ascontiguousarray(image, dtype=np.int_)
        distance = np.ascontiguousarray(distance)
        gc_saved = gc.get_threshold()
        gc.set_threshold(0)
        (
            nodes,
            directed_edges_list,
            undirected_edges_list,
            undirected_edges,
        ) = make_graph(
            center_index, len(center_index), neighbor_index, image, distance, n_atoms
        )
        
        graph = Graph(nodes=nodes)
        graph.directed_edges_list = directed_edges_list
        graph.undirected_edges_list = undirected_edges_list
        graph.undirected_edges = undirected_edges
        gc.set_threshold(gc_saved[0])
        
        return graph

    def as_dict(self) -> dict[str, float]:
        """Save the args of the graph converter."""
        return {
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "line_graph_cutoff": self.line_graph_cutoff,
            "algorithm": self.algorithm,
            "neighbor_backend": self.neighbor_backend,
            "neighbor_device": self.neighbor_device,
            "neighbor_method": self.neighbor_method,
        }

    @classmethod
    def from_dict(cls, dict) -> GraphConverter:
        """Create converter from dictionary."""
        kwargs = dict.copy()
        kwargs.pop("algorithm", None)
        return GraphConverter(**kwargs)
