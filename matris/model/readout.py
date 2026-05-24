from __future__ import annotations

import os
import time
import torch
from torch import Tensor, nn
from typing import  Literal, Union, Sequence

from .functions import (
    MLP, 
    GatedMLP,
    SwishLayer,
    GraphPooling,
    reshape_stress,
    Sphere,
    aggregate,
    get_activation,
    _load_matris_op,
)


def _readout_stage_profile_enabled() -> bool:
    return (
        os.environ.get("MATRIS_MODEL_STAGE_PROFILE", "0") == "1"
        or os.environ.get("MATRIS_CALCULATOR_STAGE_PROFILE", "0") == "1"
    )


def _sync_if_needed_from_tensors(*values) -> None:
    if not torch.cuda.is_available():
        return
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            torch.cuda.synchronize()
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor) and item.is_cuda:
                    torch.cuda.synchronize()
                    return


def _timed_readout_stage(profile: dict[str, float], name: str, sync_values: tuple, fn):
    if not _readout_stage_profile_enabled():
        return fn()
    _sync_if_needed_from_tensors(*sync_values)
    start = time.perf_counter()
    result = fn()
    _sync_if_needed_from_tensors(result, *sync_values)
    profile[name] = (time.perf_counter() - start) * 1000.0
    return result


def _use_p72_geom_force_stress() -> bool:
    return os.environ.get("MATRIS_P72_GEOM_FORCE_STRESS", "0") == "1"


def _p72_geom_backend() -> str:
    return os.environ.get("MATRIS_P72_GEOM_BACKEND", "python").strip().lower()


def _p72_geom_inputs() -> str:
    return os.environ.get("MATRIS_P72_GEOM_INPUTS", "edge").strip().lower()


def _zero_like(value: Tensor) -> Tensor:
    return value.detach().new_zeros(value.shape)


def _p72_grad_edge_from_length_unit(
    batch_graph,
    grad_lengths: Tensor | None,
    grad_unit_vectors: Tensor | None,
) -> Tensor:
    edge_lengths = batch_graph["batch_edge_lengths"].detach()
    unit_edge_vectors = batch_graph["batch_unit_edge_vectors"].detach()
    if grad_lengths is None:
        grad_lengths = _zero_like(edge_lengths)
    if grad_unit_vectors is None:
        grad_unit_vectors = _zero_like(unit_edge_vectors)
    grad_lengths = grad_lengths.to(dtype=unit_edge_vectors.dtype)
    grad_unit_vectors = grad_unit_vectors.to(dtype=unit_edge_vectors.dtype)
    parallel = (grad_unit_vectors * unit_edge_vectors).sum(dim=1, keepdim=True)
    grad_unit_to_edge = (grad_unit_vectors - parallel * unit_edge_vectors) / edge_lengths[:, None]
    return grad_lengths[:, None] * unit_edge_vectors + grad_unit_to_edge


def _p72_force_stress_from_edge_python(
    batch_graph,
    grad_edge_vectors: Tensor,
    compute_stress: bool,
) -> tuple[Tensor, Tensor | None]:
    target_index = batch_graph["atom_graph_dict"]["target_index"]
    source_index = batch_graph["atom_graph_dict"]["source_index"]
    num_atoms = int(batch_graph["batch_cart_coords"].shape[0])
    force = grad_edge_vectors.new_zeros((num_atoms, 3))
    force.index_add_(0, target_index, -grad_edge_vectors)
    force.index_add_(0, source_index, grad_edge_vectors)

    if not compute_stress:
        return force, None

    edge_vectors = batch_graph["batch_edge_vectors"].detach().to(dtype=grad_edge_vectors.dtype)
    atom_segment = batch_graph["atom_segment"]
    edge_segment = atom_segment.index_select(0, target_index)
    volumes = batch_graph["volumes"].to(dtype=grad_edge_vectors.dtype)
    raw_stress = grad_edge_vectors.new_zeros((int(volumes.shape[0]), 3, 3))
    outer = 0.5 * (
        edge_vectors[:, :, None] * grad_edge_vectors[:, None, :]
        + edge_vectors[:, None, :] * grad_edge_vectors[:, :, None]
    )
    raw_stress.index_add_(0, edge_segment, outer)
    stress = raw_stress * (1.0 / volumes * 160.21766208)
    return force, stress


def _p72_force_stress_from_edge_cuda(
    batch_graph,
    grad_edge_vectors: Tensor,
) -> tuple[Tensor, Tensor] | None:
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "force_stress_from_edge_vectors"):
        return None
    if (
        not grad_edge_vectors.is_cuda
        or grad_edge_vectors.dtype != torch.float32
        or "batch_edge_vectors" not in batch_graph
    ):
        return None
    edge_vectors = batch_graph["batch_edge_vectors"].detach()
    target_index = batch_graph["atom_graph_dict"]["target_index"]
    source_index = batch_graph["atom_graph_dict"]["source_index"]
    atom_segment = batch_graph["atom_segment"]
    volumes = batch_graph["volumes"]
    if not (
        edge_vectors.is_cuda
        and target_index.is_cuda
        and source_index.is_cuda
        and atom_segment.is_cuda
        and volumes.is_cuda
        and edge_vectors.dtype == torch.float32
        and volumes.dtype == torch.float32
        and target_index.dtype == torch.int64
        and source_index.dtype == torch.int64
        and atom_segment.dtype == torch.int64
    ):
        return None
    force, stress = matris_op.force_stress_from_edge_vectors(
        grad_edge_vectors.contiguous(),
        edge_vectors.contiguous(),
        target_index.contiguous(),
        source_index.contiguous(),
        atom_segment.contiguous(),
        volumes.contiguous(),
    )
    return force, stress


def _p72_force_stress_from_edge(
    batch_graph,
    grad_edge_vectors: Tensor,
    compute_stress: bool,
) -> tuple[Tensor, Tensor | None]:
    if _p72_geom_backend() == "cuda":
        cuda_result = _p72_force_stress_from_edge_cuda(batch_graph, grad_edge_vectors)
        if cuda_result is not None:
            force, stress = cuda_result
            return force, stress if compute_stress else None
    return _p72_force_stress_from_edge_python(
        batch_graph,
        grad_edge_vectors,
        compute_stress=compute_stress,
    )


def _can_use_p72_geom_force_stress(batch_graph, compute_force: bool, is_training: bool) -> bool:
    if not (_use_p72_geom_force_stress() and compute_force and not is_training):
        return False
    if "batch_edge_vectors" not in batch_graph:
        return False
    if _p72_geom_inputs() == "length_unit":
        return (
            "batch_edge_lengths" in batch_graph
            and "batch_unit_edge_vectors" in batch_graph
        )
    return True

class EnergyHead(nn.Module):
    def __init__(
        self,
        feat_dim: int = 128,
        hidden_dim: Union[int, Sequence[int]] = 128,
        output_dim: int = 1,
        mlp_type: str = "GateMLP",
        activation_type: str = "silu",
    ):
        """
        Args:
            feat_dim : Dimension of the input node (atom) features.
            hidden_dim : Hidden layer size(s).
            output_dim : Output dimension (usually 1 for energy).
            mlp_type : Type of MLP to use: "MLP", "GateMLP" or "MOE".
            activation_type : Activation function to use: "silu", "relu", "tanh", "gelu".
        """
        super().__init__()

        # Ensure hidden_dim is a list of ints
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if mlp_type.lower() == "mlp":
            self.energy_head = MLP(
                input_dim=feat_dim,
                hidden_dim=hidden_dims,
                output_dim=output_dim,
                activation=activation_type,
            )
        elif mlp_type.lower() == "gatemlp":
            self.energy_head = nn.Sequential(
                GatedMLP(
                    input_dim=feat_dim,
                    hidden_dim=hidden_dims,
                    output_dim=hidden_dims[-1],
                    norm_type="layer",
                    activation=activation_type,
                ),
                nn.Linear(in_features=hidden_dims[-1], out_features=output_dim),
            )
        else:
            raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")

        self.pooling = GraphPooling(average=False)

    def forward(self, batch_graph, node_feat: Tensor) -> Tensor:
        """Compute the total energy per graph/batch from atomic features.

        Args:
            atom_feat : Tensor(N_atoms, feat_dim) – atomic feature vectors.
            batch_index : Tensor(N_atoms,) – graph index for each atom.
        """
        energies = self.energy_head(node_feat)  # (N_atoms, output_dim)
        total_energy = self.pooling(energies, batch_graph['atom_segment']).view(-1)  # (N_graphs,)
        return total_energy

class MagmomHead(nn.Module):
    def __init__(
        self,
        feat_dim: int = 128,
        hidden_dim: Union[int, Sequence[int]] = 128,
        output_dim: int = 1,
        mlp_type: str = "GateMLP",
        activation_type: Literal["silu", "relu", "tanh", "gelu"] = "silu",
    ):
        """
        Args:
            feat_dim : Dimension of the input node (atom) features.
            hidden_dim : Hidden layer size(s).
            output_dim : Output dimension.
            mlp_type : Type of MLP to use: "MLP", "GateMLP" or "MOE".
            activation_type : Activation function to use: "silu", "relu", "tanh", "gelu".
        """
        super().__init__()
        
        # Ensure hidden_dim is a list of ints
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if mlp_type.lower() == "mlp":
            self.magmom_head = MLP(
                input_dim=feat_dim,
                hidden_dim=hidden_dims,
                output_dim=output_dim,
                activation=activation_type,
            )
        elif mlp_type.lower() == "gatemlp":
            self.magmom_head = nn.Sequential(
                GatedMLP(
                    input_dim=feat_dim,
                    hidden_dim=hidden_dims,
                    output_dim=hidden_dims[-1],
                    norm_type="layer",
                    activation=activation_type,
                ),
                nn.Linear(in_features=hidden_dims[-1], out_features=output_dim),
            )
        else:
            raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")

    def forward(self, batch_graph, node_feat: Tensor) -> Tensor:
        """Compute the magmom per atom from atomic features.
        
        Args:
            atom_feat : Tensor,
            batch_index : List(N_graphs,) - atom numbers for per graph.
        """

        magmom_feat = torch.abs(self.magmom_head(node_feat)) # [N_atoms, 1]
        magmoms = torch.split(magmom_feat.view(-1), batch_graph['atoms_per_graph']) # Tuple
        return list(magmoms) # [N_atoms]


class ForceStressHead(nn.Module):
    def __init__(
        self,
        is_conservation: bool = True,
        feat_dim: int = 128,
        hidden_dim: Union[int, Sequence[int]] = 128,
        output_dim: int = 3,
        mlp_type: str = "GateMLP",
        activation_type: str = "silu",
    ):
        """
        Args:
            is_conservation (bool): using 'direct' or 'autograd(cons)' method.
            feat_dim (int): edge feat dim.
            hidden_dim (int): MLP hidden dim.
            output_dim (int): output dim, force label should be 3.
            mlp_type (str): MLP type.
                Can be "MLP", "GateMLP".
            activation_type (str): activation type.
                Can be "SiLU", "Sigmoid"... See fucntion.py for more informations.
        """
        super().__init__()
        self.is_conservation = is_conservation
        self.last_profile: dict[str, float] = {}
        # Ensure hidden_dim is a list of ints
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)
            
        if not self.is_conservation:
            self.stress_head = EquivariantHead(
                feas_dim=feat_dim,
                lmax=2,
                use_bias=False,
                activation=activation_type,
            )
            if mlp_type.lower() == "mlp":
                self.force_head = MLP(
                    input_dim=feat_dim,
                    hidden_dim=hidden_dims,
                    output_dim=output_dim,
                    activation=activation_type,
                )
            elif mlp_type.lower() == "gatemlp":
                self.force_head = nn.Sequential(
                    GatedMLP(
                        input_dim=feat_dim,
                        hidden_dim=hidden_dims,
                        output_dim=hidden_dims[-1],
                        norm_type="layer",
                        activation=activation_type,
                    ),
                    nn.Linear(in_features=hidden_dims[-1], out_features=output_dim),
                )
            else:
                raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")
            
    def forward(self, 
                batch_graph,
                compute_force: bool = True, 
                compute_stress: bool = True,
                total_energy: Tensor = None, 
                node_feat: Tensor = None, 
                edge_feat: Tensor = None, 
                is_training: bool=False):
        predict = {}
        profile: dict[str, float] = {}
        if self.is_conservation:
            assert total_energy is not None
            if _can_use_p72_geom_force_stress(batch_graph, compute_force, is_training):
                if _p72_geom_inputs() == "length_unit":
                    geom_grads = _timed_readout_stage(
                        profile,
                        "autograd_grad_ms",
                        (total_energy,),
                        lambda: torch.autograd.grad(
                            total_energy.sum(),
                            [
                                batch_graph["batch_edge_lengths"],
                                batch_graph["batch_unit_edge_vectors"],
                            ],
                            create_graph=False,
                            retain_graph=False,
                            allow_unused=True,
                        ),
                    )
                else:
                    geom_grads = _timed_readout_stage(
                        profile,
                        "autograd_grad_ms",
                        (total_energy,),
                        lambda: (
                            torch.autograd.grad(
                                total_energy.sum(),
                                [batch_graph["batch_edge_vectors"]],
                                create_graph=False,
                                retain_graph=False,
                            )[0],
                        ),
                    )

                def postprocess_p72_force_stress() -> None:
                    if _p72_geom_inputs() == "length_unit":
                        grad_edge_vectors = _p72_grad_edge_from_length_unit(
                            batch_graph,
                            geom_grads[0],
                            geom_grads[1],
                        )
                    else:
                        grad_edge_vectors = geom_grads[0]
                    force, stress = _p72_force_stress_from_edge(
                        batch_graph,
                        grad_edge_vectors,
                        compute_stress=compute_stress,
                    )
                    predict["f"] = torch.split(force, batch_graph["atoms_per_graph"])
                    if compute_stress and stress is not None:
                        predict["s"] = list(torch.unbind(stress, dim=0))

                _timed_readout_stage(
                    profile,
                    "force_stress_postprocess_ms",
                    geom_grads,
                    postprocess_p72_force_stress,
                )
                if _readout_stage_profile_enabled():
                    self.last_profile = profile
                return predict
            if compute_force and compute_stress:
                grad = _timed_readout_stage(
                    profile,
                    "autograd_grad_ms",
                    (total_energy,),
                    lambda: torch.autograd.grad(
                        total_energy.sum(), [batch_graph['batch_cart_coords'], batch_graph['batch_strains']], 
                        create_graph=is_training, retain_graph=is_training
                    ),
                )
                def postprocess_force_stress() -> None:
                    # force
                    force = -1 * grad[0]
                    predict["f"] = torch.split(force, batch_graph['atoms_per_graph'])
                    # stress 
                    stress = grad[1]
                    scale = 1 / batch_graph['volumes'] * 160.21766208 # units.GPa
                    stress = stress * scale
                    stress = list(torch.unbind(stress, dim=0))
                    predict["s"] = stress

                _timed_readout_stage(
                    profile,
                    "force_stress_postprocess_ms",
                    grad,
                    postprocess_force_stress,
                )
            elif compute_force:
                grad = _timed_readout_stage(
                    profile,
                    "autograd_grad_ms",
                    (total_energy,),
                    lambda: torch.autograd.grad(
                        total_energy.sum(), [batch_graph['batch_cart_coords']], 
                        create_graph=is_training, retain_graph=is_training
                    ),
                ) 
                def postprocess_force() -> None:
                    force = -1 * grad[0]
                    predict["f"] = torch.split(force, batch_graph['atoms_per_graph'])

                _timed_readout_stage(
                    profile,
                    "force_stress_postprocess_ms",
                    grad,
                    postprocess_force,
                )
        else:
            if compute_stress:
                assert node_feat is not None
                def direct_stress() -> None:
                    atom_coord = batch_graph['batch_cart_coords'] / ((torch.norm(batch_graph['batch_cart_coords'], dim=1)*(1-1e-6) + 1e-6).unsqueeze(1))
                    stress = self.stress_head(node_feat, atom_coord) # [atoms, 1+3+5]
                    stress = aggregate(stress,  batch_graph['atom_segment'], average=True) #[graphs , 1+3+5]
                    
                    # Reshape 
                    isotropic = stress[:, 0] # L=0
                    anisotropic = stress[:, [4,5,6,7,8]] # L=2
                    stress = reshape_stress(isotropic, anisotropic, stress.shape[0]) #[graphs, 3, 3]
                    scale = (1 / batch_graph['volumes'] * 160.21766208)
                    stress = stress * scale
                    stress = list(torch.unbind(stress, dim=0))
                    predict["s"] = stress

                _timed_readout_stage(
                    profile,
                    "direct_stress_ms",
                    (node_feat,),
                    direct_stress,
                )
            if compute_force:
                assert edge_feat is not None
                def direct_force() -> None:
                    direct_edge_feas = torch.index_select(edge_feat, 0, batch_graph['directed2undirected'])
                    direct_edge_feas = self.force_head(direct_edge_feas)
                    direct_edge_feas = direct_edge_feas * batch_graph['unit_edge_vectors']

                    force = aggregate(
                        direct_edge_feas, batch_graph['atom_graph_dict']['target_index'],
                        average=False, 
                        num_segment=len(node_feat)
                    ) # [N_atoms, 3]
                    predict["f"] = torch.split(force, batch_graph['atoms_per_graph'])

                _timed_readout_stage(
                    profile,
                    "direct_force_ms",
                    (edge_feat,),
                    direct_force,
                )
        
        if _readout_stage_profile_enabled():
            self.last_profile = profile
        return predict 


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        activation="silu",
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        
        self.act = get_activation(activation)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, out_channels*2),
        )

    def forward(self, feas, vec_feas):
        # feas: [N_atoms, dim]
        # vec_feas: [N_atoms, 3, dim]
        vec1 = torch.norm(self.vec1_proj(vec_feas), dim=-2) # [N_atoms, dim]
         
        vec2 = self.vec2_proj(vec_feas)
        
        feas = torch.cat([feas, vec1], dim=-1)
        feas, vec_feas = torch.split(self.update_net(feas), self.out_channels, dim=-1)
        vec_feas = vec_feas.unsqueeze(1) * vec2
        
        if self.act is not None:
            feas = self.act(feas)
        return feas, vec_feas

class EquivariantHead(nn.Module):
    def __init__(
        self, 
        feas_dim: int = 64,
        lmax: int = 1,
        use_bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        assert lmax<=2, "Only support lmax <= 2"
        
        self.sh = Sphere(lmax = lmax)
        self.feas_wise_linear = nn.Linear(feas_dim, feas_dim, bias=False)
        self.basis_wise_linear = nn.Linear(1, feas_dim, bias=False)
        
        self.equivariant_block = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    feas_dim,
                    feas_dim// 2,
                    activation=activation,
                ),
                GatedEquivariantBlock(feas_dim // 2, 1, activation=activation),
            ]
        )
    
    def forward(self, feas, vec):
        # feas: [N_atoms, dim]
        # vec: [N_atoms, 3]
        sh_basis = self.sh(vec) # [N_atoms, ( (lmax+1) ** 2) - 1]
        # reshape: [edges, (lmax+1)**2 - 1] -> [N_atoms, (lmax+1)**2 - 1, 1]
        sh_basis = sh_basis.unsqueeze(2)
        vec_feas = self.basis_wise_linear(sh_basis) #[N_atoms, 3, dim]
        
        feas = self.feas_wise_linear(feas) # [N_atoms, dim]
        
        for layer in self.equivariant_block:
            feas, vec_feas = layer(feas, vec_feas)
        
        return vec_feas.squeeze() #[N_atoms, 3]
