from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes, all_properties
import numpy as np
import os
import time
import torch

from ..model.model import MatRIS

from pymatgen.io.ase import AseAtomsAdaptor
from ..graph import RadiusGraph

from ase.optimize import (
    BFGS, BFGSLineSearch, 
    FIRE, LBFGS, 
    LBFGSLineSearch, MDMin
)

names = [
    "BFGS", "BFGSLineSearch", 
    "FIRE", "LBFGS", 
    "LBFGSLineSearch", "MDMin"
]

OPTIMIZERS = {name: globals()[name] for name in names}


def _calculator_stage_profile_enabled() -> bool:
    return os.environ.get("MATRIS_CALCULATOR_STAGE_PROFILE", "0") == "1"


def _sync_if_needed(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_stage(device: str, profile: dict[str, float], name: str, fn):
    if not _calculator_stage_profile_enabled():
        return fn()
    _sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    _sync_if_needed(device)
    profile[name] = (time.perf_counter() - start) * 1000.0
    return result


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        export_value = value.detach()
        if export_value.dtype == torch.bfloat16:
            export_value = export_value.to(torch.float32)
        return export_value.cpu().numpy()
    return value

class MatRISCalculator(Calculator):
    """MatRIS Calculator for ASE applications."""
    
    implemented_properties = ("energy", "forces", "stress", "magmoms")  # type: ignore
    
    def __init__(
        self,
        model: str = "matris_10m_oam",
        task: str = "efs",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """
        Args:
            model (MatRIS): Instance of a MatRIS model. If set to None, the default MatRIS is loaded.
            task (str): The prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
            device (str): The device to be used for predictions,
            stress_unit (float): the conversion factor to convert GPa(MatRIS default) to eV/A^3.
            **kwargs: Passed to the Calculator parent class.
        """
        super().__init__(**kwargs)
        self.task=task
        self.device = device
        self.model = MatRIS.load(model_name=model, device=self.device)
        
        self.stress_unit = units.GPa
        key = ["atoms_per_graph", "ref_energy"]
        for t in task:
            key.append(t)
        self.key = set(key)
        self.last_profile: dict[str, float] = {}

    def _adjust_pbc(self, atoms: Atoms) -> None:
        pbc = atoms.get_pbc()
        if pbc[0] and pbc[1] and pbc[2]:
            return

        pos = atoms.get_positions()
        cell = np.array(atoms.get_cell())

        identity = np.identity(3, dtype=float)
        max_positions = np.max(np.absolute(pos)) + 1

        cutoff = self.model.config["pairwise_cutoff"]
        expand = max(5, self.model.config["num_layers"])

        if not pbc[0]:
            cell[0, :] = max_positions * expand * cutoff * identity[0, :]
        if not pbc[1]:
            cell[1, :] = max_positions * expand * cutoff * identity[1, :]
        if not pbc[2]:
            cell[2, :] = max_positions * expand * cutoff * identity[2, :]

        atoms.set_cell(cell, scale_atoms=False)

    def _export_prediction_item(
        self,
        model_prediction: dict,
        item_idx: int,
        n_atoms: float,
    ) -> dict:
        result = {}
        ref_energy = model_prediction.get("ref_energy", 0)
        if isinstance(ref_energy, torch.Tensor):
            ref_energy_item = ref_energy[item_idx] if ref_energy.ndim > 0 else ref_energy
        else:
            ref_energy_item = ref_energy

        energy_item = model_prediction["e"][item_idx]
        result["ref_energy"] = _to_numpy(ref_energy_item) * n_atoms
        result["energy"] = _to_numpy(energy_item) * n_atoms

        if "f" in model_prediction:
            result["forces"] = _to_numpy(model_prediction["f"][item_idx])
        else:
            result["forces"] = None

        if "s" in model_prediction:
            result["stress"] = _to_numpy(model_prediction["s"][item_idx]) * self.stress_unit
        else:
            result["stress"] = None

        if "m" in model_prediction:
            result["magmoms"] = _to_numpy(model_prediction["m"][item_idx])
        else:
            result["magmoms"] = None

        return result

    def calculate_many(self, atoms_list: list[Atoms]) -> list[dict]:
        """Evaluate multiple structures in one model forward.

        This is an opt-in scheduling fast path for offline inference. It does
        not populate ASE's per-Atoms calculator cache and leaves calculate()
        semantics unchanged.
        """
        if not atoms_list:
            return []

        graphs = []
        n_atoms_list = []
        for atoms in atoms_list:
            self._adjust_pbc(atoms)
            structure = AseAtomsAdaptor.get_structure(atoms)
            graph_cpu = self.model.graph_converter(structure)
            graphs.extend([graph_cpu] if isinstance(graph_cpu, RadiusGraph) else graph_cpu)
            n_atoms_list.append(
                1 if not self.model.is_intensive else structure.composition.num_atoms
            )

        return self.calculate_graphs(graphs, n_atoms_list)

    def calculate_graphs(self, graphs: list[RadiusGraph], n_atoms_list: list[float]) -> list[dict]:
        """Evaluate prebuilt graphs.

        CPU-side graph construction can be scheduled outside this call, while
        this method keeps the model/device/export semantics shared with
        calculate_many().
        """
        device_graphs = []
        for graph in graphs:
            graph_on_device = graph.to(self.device)
            device_graphs.extend([graph_on_device] if isinstance(graph_on_device, RadiusGraph) else graph_on_device)

        model_prediction = self.model(
            device_graphs,
            task=self.task,
            is_training=False,
        )
        return [
            self._export_prediction_item(model_prediction, idx, n_atoms)
            for idx, n_atoms in enumerate(n_atoms_list)
        ]
     
    def calculate(
        self,
        atoms: Atoms,
        properties: list,
        system_changes: list,
    ) -> None:
        
        profile_enabled = _calculator_stage_profile_enabled()
        profile: dict[str, float] = {}
        total_start = time.perf_counter() if profile_enabled else 0.0
        properties = properties or all_properties
        system_changes = system_changes or all_changes

        _timed_stage(
            self.device,
            profile,
            "ase_super_calculate_ms",
            lambda: Calculator.calculate(
                self,
                atoms=atoms,
                properties=properties,
                system_changes=system_changes,
            ),
        )

        _timed_stage(self.device, profile, "pbc_adjust_ms", lambda: self._adjust_pbc(atoms))

        structure = _timed_stage(
            self.device,
            profile,
            "ase_atoms_to_structure_ms",
            lambda: AseAtomsAdaptor.get_structure(atoms),
        )
        graph_cpu = _timed_stage(
            self.device,
            profile,
            "graph_converter_ms",
            lambda: self.model.graph_converter(structure),
        )
        graph = _timed_stage(
            self.device,
            profile,
            "graph_to_device_ms",
            lambda: graph_cpu.to(self.device),
        )
        
        graphs = [graph] if isinstance(graph, RadiusGraph) else graph
        
        model_prediction = _timed_stage(
            self.device,
            profile,
            "model_forward_total_ms",
            lambda: self.model(
                graphs,
                task = self.task,
                is_training = False,
            ),
        )
        if profile_enabled:
            for key, value in getattr(self.model, "last_forward_profile", {}).items():
                if isinstance(value, (int, float)):
                    profile[f"model.{key}"] = float(value)

        n_atoms = 1 if not self.model.is_intensive else structure.composition.num_atoms
        model_predictions = _timed_stage(
            self.device,
            profile,
            "tensor_cpu_numpy_export_ms",
            lambda: self._export_prediction_item(model_prediction, 0, n_atoms),
        )
        
        # Convert Result
        def update_results() -> None:
            self.results.update(
                ref_energy=model_predictions["ref_energy"],
                energy=model_predictions["energy"],
                forces=model_predictions.get("forces", None),
                stress=model_predictions.get("stress", None),
                magmoms=model_predictions.get("magmoms", None),
            )

        _timed_stage(self.device, profile, "results_update_ms", update_results)
        if profile_enabled:
            profile["calculator_calculate_total_ms"] = (time.perf_counter() - total_start) * 1000.0
            profile["n_atoms"] = float(len(atoms))
            self.last_profile = profile


class TrajectoryObserver:
    # ref: https://github.com/CederGroupHub/chgnet

    def __init__(self, atoms: Atoms) -> None:
        
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class CrystalFeasObserver:
    # ref: https://github.com/CederGroupHub/chgnet

    def __init__(self, atoms: Atoms) -> None:
        """Create a CrystalFeasObserver from an Atoms object."""
        self.atoms = atoms
        self.crystal_feature_vectors: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record Atoms crystal feature vectors after an MD/relaxation step."""
        self.crystal_feature_vectors.append(self.atoms._calc.results["crystal_fea"])

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.crystal_feature_vectors)

    def save(self, filename: str) -> None:
        """Save the crystal feature vectors to filename in pickle format."""
        out_pkl = {"crystal_feas": self.crystal_feature_vectors}
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)
                      
