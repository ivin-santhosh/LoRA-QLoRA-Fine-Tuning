import math
import unittest
from modelstack_lab.benchmark import graph_benchmark, summarize_ns, timed


class BenchmarkTests(unittest.TestCase):
    def test_graph_reproducibility(self):
        one = graph_benchmark(size=2, iterations=2, seed=11)
        two = graph_benchmark(size=2, iterations=2, seed=11)
        self.assertEqual(one["verified_max_abs_error"], 0)
        self.assertEqual(two["verified_max_abs_error"], 0)
        self.assertEqual(one["eager"]["intermediate_output_elements"], 12)
        self.assertEqual(one["fused"]["intermediate_output_elements"], 4)

    def test_reject_invalid_bounds(self):
        for size in (0, 97, -3):
            with self.assertRaises(ValueError):
                graph_benchmark(size=size, iterations=2)
        with self.assertRaises(ValueError):
            timed(lambda: None, warmup=0, iterations=2)
        with self.assertRaises(ValueError):
            timed(lambda: None, warmup=1, iterations=1)

    def test_p95(self):
        value = summarize_ns([1000000, 2000000, 3000000, 4000000])
        self.assertEqual(value["p95_ms"], 4)
        self.assertEqual(value["median_ms"], 2.5)
        self.assertTrue(math.isfinite(value["median_ms"]))


if __name__ == "__main__":
    unittest.main()
