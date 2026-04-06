from __future__ import annotations

import unittest
import numpy as np

from regime_toolkit.indicators import (
    effective_dimension_velocity_pca,
    velocity_participation_ratio_time_weighted,
    velocity_participation_ratio_arclength_weighted,
    occupancy_gap,
    speed_heterogeneity,
    rosenstein_style_divergence_rate,
)


class TestIndicators(unittest.TestCase):
    def test_straight_line_has_one_dominant_direction(self) -> None:
        t = np.linspace(0.0, 10.0, 201)
        x = np.column_stack([t, np.zeros_like(t)])

        time_pr = velocity_participation_ratio_time_weighted(t, x)
        arc_pr = velocity_participation_ratio_arclength_weighted(t, x)
        gap = occupancy_gap(t, x)
        speed = speed_heterogeneity(t, x, metric="cv")
        eff_dim = effective_dimension_velocity_pca(
            t, x, metric="participation_ratio", demean=False
        )

        self.assertAlmostEqual(time_pr.value, 1.0, places=6)
        self.assertAlmostEqual(arc_pr.value, 1.0, places=6)
        self.assertAlmostEqual(gap, 0.0, places=6)
        self.assertAlmostEqual(speed.value, 0.0, places=6)
        self.assertAlmostEqual(eff_dim.dimension, 1.0, places=6)

    def test_rosenstein_divergence_is_finite_on_chaotic_scalar_series(self) -> None:
        x = 0.123456789
        series = []
        for n in range(3500):
            x = 4.0 * x * (1.0 - x)
            if n >= 500:
                series.append(x)
        y = np.asarray(series, dtype=float)

        result = rosenstein_style_divergence_rate(
            y,
            dt=1.0,
            embedding_dim=3,
            delay=1,
            theiler_window=20,
            max_horizon_steps=12,
            fit_start_step=1,
            fit_stop_step=5,
            min_pairs_per_step=20,
        )

        self.assertTrue(np.isfinite(result.rate))
        self.assertGreater(result.rate, 0.0)
        self.assertGreater(result.rvalue, 0.9)
        self.assertTrue(np.all(result.counts[:5] >= 20))


if __name__ == "__main__":
    unittest.main()
