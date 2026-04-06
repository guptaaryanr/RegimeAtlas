from __future__ import annotations

import unittest
import numpy as np

from regime_toolkit.observations import (
    state_to_scalar_observation,
    add_observation_noise,
    delay_embed,
    observed_trajectory,
)


class TestObservations(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=float,
        )

    def test_coordinate_linear_and_norm_observations(self) -> None:
        y_coord = state_to_scalar_observation(self.x, mode="coordinate", index=1)
        np.testing.assert_allclose(y_coord, np.array([2.0, 4.0, 6.0, 8.0]))

        y_linear = state_to_scalar_observation(
            self.x, mode="linear", weights=np.array([2.0, -1.0])
        )
        np.testing.assert_allclose(y_linear, np.array([0.0, 2.0, 4.0, 6.0]))

        y_norm = state_to_scalar_observation(self.x, mode="norm")
        np.testing.assert_allclose(y_norm, np.linalg.norm(self.x, axis=1))

    def test_noise_is_reproducible_and_zero_sigma_is_identity(self) -> None:
        y = np.linspace(0.0, 1.0, 10)
        y0 = add_observation_noise(y, sigma=0.0, seed=123)
        np.testing.assert_allclose(y0, y)

        y1 = add_observation_noise(y, sigma=0.1, seed=123)
        y2 = add_observation_noise(y, sigma=0.1, seed=123)
        np.testing.assert_allclose(y1, y2)
        self.assertFalse(np.allclose(y1, y))

    def test_delay_embed_shape_and_anchor_indices(self) -> None:
        y = np.arange(10, dtype=float)
        z, anchors = delay_embed(y, embedding_dim=3, delay=2, return_time_indices=True)
        self.assertEqual(z.shape, (6, 3))
        np.testing.assert_array_equal(z[0], np.array([0.0, 2.0, 4.0]))
        np.testing.assert_array_equal(z[-1], np.array([5.0, 7.0, 9.0]))
        np.testing.assert_array_equal(anchors, np.array([4, 5, 6, 7, 8, 9]))

    def test_observed_trajectory_full_state_and_delay_scalar(self) -> None:
        obs_full = observed_trajectory(self.x, representation="full_state")
        self.assertEqual(obs_full.data.shape, self.x.shape)
        np.testing.assert_array_equal(obs_full.time_indices, np.arange(self.x.shape[0]))

        obs_delay = observed_trajectory(
            self.x,
            representation="delay_scalar",
            scalar_mode="coordinate",
            index=0,
            embedding_dim=2,
            delay=1,
        )
        self.assertEqual(obs_delay.data.shape, (3, 2))
        np.testing.assert_array_equal(obs_delay.time_indices, np.array([1, 2, 3]))
        np.testing.assert_allclose(obs_delay.data[0], np.array([1.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
