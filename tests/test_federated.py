import torch

from safu.federated import ClientUpdate, fedavg, state_bytes


def test_fedavg_weights_by_sample_count_and_keeps_integer_buffer():
    first = ClientUpdate(0, 1, {"weight": torch.tensor([1.0]), "count": torch.tensor(2)}, 0.0, 0.0)
    second = ClientUpdate(1, 3, {"weight": torch.tensor([5.0]), "count": torch.tensor(4)}, 0.0, 0.0)
    state = fedavg([first, second])
    assert torch.allclose(state["weight"], torch.tensor([4.0]))
    assert state["count"].item() == 4
    assert state_bytes(state) == 12
