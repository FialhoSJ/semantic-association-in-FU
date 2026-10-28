import numpy as np
import pytest

from safu.data import dirichlet_partition


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_dirichlet_partition_is_complete_disjoint_and_reproducible(alpha):
    labels = np.repeat(np.arange(10), 50)
    left = dirichlet_partition(labels, 8, alpha, seed=17, min_samples=3)
    right = dirichlet_partition(labels, 8, alpha, seed=17, min_samples=3)
    assert all(np.array_equal(a, b) for a, b in zip(left, right))
    assert all(len(part) >= 3 for part in left)
    assert np.array_equal(np.sort(np.concatenate(left)), np.arange(len(labels)))


def test_partition_rejects_impossible_minimum():
    with pytest.raises(ValueError, match="Not enough"):
        dirichlet_partition(np.arange(5), 3, 0.5, 1, min_samples=2)
