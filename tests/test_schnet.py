import pytest
import torch
from torch_geometric.data import Batch, Data

from reg_transfo.algorithms.schnet import MoleculeSchNet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cremp_batch():
    """CREMP-style: 5-col one-hot (H, C, N, O, S), returns (batch_graph, images, energies)."""
    x1 = torch.zeros(3, 5)
    x1[0, 0] = 1.0  # H
    x1[1, 1] = 1.0  # C
    x1[2, 2] = 1.0  # N

    x2 = torch.zeros(2, 5)
    x2[0, 3] = 1.0  # O
    x2[1, 4] = 1.0  # S

    d1 = Data(x=x1, pos=torch.rand(3, 3), y=torch.tensor([1.0]))
    d2 = Data(x=x2, pos=torch.rand(2, 3), y=torch.tensor([2.0]))
    batch_graph = Batch.from_data_list([d1, d2])

    batch_images = torch.rand(2, 1, 32, 32)
    batch_energies = torch.tensor([1.0, 2.0])
    return batch_graph, batch_images, batch_energies


def _make_dchem_batch():
    """DeepChem-style: 8-col one-hot (H, C, N, O, F, P, S, Cl)."""
    x1 = torch.zeros(3, 8)
    x1[0, 0] = 1.0  # H  → z=1
    x1[1, 1] = 1.0  # C  → z=6
    x1[2, 4] = 1.0  # F  → z=9

    x2 = torch.zeros(2, 8)
    x2[0, 6] = 1.0  # S  → z=16
    x2[1, 7] = 1.0  # Cl → z=17

    d1 = Data(x=x1, pos=torch.rand(3, 3), y=torch.tensor([1.0]))
    d2 = Data(x=x2, pos=torch.rand(2, 3), y=torch.tensor([2.0]))
    return Batch.from_data_list([d1, d2])


def _make_qm7b_batch():
    """QM7b-style: no x, but z (atomic numbers) is present."""
    d1 = Data(z=torch.tensor([6, 1, 1]), pos=torch.rand(3, 3), y=torch.tensor([1.0]))
    d2 = Data(z=torch.tensor([8, 7]), pos=torch.rand(2, 3), y=torch.tensor([2.0]))
    return Batch.from_data_list([d1, d2])


# ---------------------------------------------------------------------------
# Tests — forward shape
# ---------------------------------------------------------------------------

def test_forward_cremp_5col():
    model = MoleculeSchNet()
    batch_graph, _, _ = _make_cremp_batch()
    out = model(batch_graph)
    assert out.shape == (2,)


def test_forward_dchem_8col():
    model = MoleculeSchNet()
    batch = _make_dchem_batch()
    out = model(batch)
    assert out.shape == (2,)


def test_forward_multitarget():
    """QM8/QM9-style: out_channels > 1."""
    model = MoleculeSchNet(out_channels=16)
    batch = _make_dchem_batch()
    out = model(batch)
    assert out.shape == (2, 16)


def test_forward_qm7b_z():
    model = MoleculeSchNet()
    batch = _make_qm7b_batch()
    out = model(batch)
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# Tests — training_step with tuple (CREMP) and plain Batch
# ---------------------------------------------------------------------------

def test_training_step_cremp_tuple():
    """CREMP collate returns (batch_graph, images, energies); SchNet must unpack."""
    model = MoleculeSchNet()
    batch_tuple = _make_cremp_batch()
    loss = model.training_step(batch_tuple, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_training_step_dchem_batch():
    model = MoleculeSchNet()
    batch = _make_dchem_batch()
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_training_step_qm7b():
    model = MoleculeSchNet()
    batch = _make_qm7b_batch()
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Tests — z mapping correctness
# ---------------------------------------------------------------------------

def test_z_mapping_cremp():
    batch_graph, _, _ = _make_cremp_batch()
    z = MoleculeSchNet._z_from_batch(batch_graph)
    assert torch.equal(z, torch.tensor([1, 6, 7, 8, 16]))


def test_z_mapping_dchem():
    batch = _make_dchem_batch()
    z = MoleculeSchNet._z_from_batch(batch)
    assert torch.equal(z, torch.tensor([1, 6, 9, 16, 17]))


def test_z_mapping_qm7b():
    batch = _make_qm7b_batch()
    z = MoleculeSchNet._z_from_batch(batch)
    assert torch.equal(z, torch.tensor([6, 1, 1, 8, 7]))


# ---------------------------------------------------------------------------
# Tests — error cases
# ---------------------------------------------------------------------------

def test_forward_no_x_no_z_raises():
    """Neither x nor z → must raise ValueError."""
    model = MoleculeSchNet()
    d = Data(pos=torch.rand(3, 3), y=torch.tensor([1.0]))
    batch = Batch.from_data_list([d])
    with pytest.raises(ValueError, match="neither 'z' nor 'x'"):
        model(batch)
