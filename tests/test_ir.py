"""Adversarial and property-style checks for the reference compiler pass."""
import random
import unittest
from modelstack_lab.ir import Node, Graph, demo_graph, execute, fuse_matmul_bias_relu


class IRTests(unittest.TestCase):
    def test_fusion_structure_and_equal_values(self):
        graph = demo_graph()
        fused = fuse_matmul_bias_relu(graph)
        self.assertEqual([n.op for n in fused.nodes], ["linear_relu"])
        for size in (1, 2, 4, 13):
            for seed in range(6):
                rng = random.Random(seed)
                feeds = {"x": [[rng.uniform(-3, 3) for _ in range(size)] for _ in range(size)],
                         "w": [[rng.uniform(-3, 3) for _ in range(size)] for _ in range(size)],
                         "b": [rng.uniform(-3, 3) for _ in range(size)]}
                a, ma = execute(graph, feeds)
                b, mb = execute(fused, feeds)
                for ra, rb in zip(a, b):
                    for aa, bb in zip(ra, rb):
                        self.assertAlmostEqual(aa, bb, places=9)
                self.assertEqual(ma["operation_nodes"], 3)
                self.assertEqual(mb["operation_nodes"], 1)

    def test_no_fusion_when_intermediate_reused(self):
        graph = Graph((Node("m", "matmul", ("x", "w")),
                       Node("a", "add_bias", ("m", "b")),
                       Node("r", "relu", ("a",)),
                       Node("d", "relu", ("m",))), "r")
        self.assertEqual(graph, fuse_matmul_bias_relu(graph))

    def test_missing_feeds(self):
        with self.assertRaisesRegex(ValueError, "missing feed"):
            execute(demo_graph(), {"x": [[1]], "w": [[1]]})

    def test_shape_failures(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            execute(demo_graph(), {"x": [[1, 2]], "w": [[1]], "b": [1]})
        with self.assertRaisesRegex(ValueError, "ragged"):
            execute(demo_graph(), {"x": [[1, 2], [3]], "w": [[1], [1]], "b": [1]})
        with self.assertRaisesRegex(ValueError, "bias"):
            execute(demo_graph(), {"x": [[1]], "w": [[1]], "b": [1, 2]})
        with self.assertRaisesRegex(ValueError, "non-finite"):
            execute(demo_graph(), {"x": [[float('nan')]], "w": [[1]], "b": [1]})

    def test_non_topological_graph_rejected(self):
        graph = Graph((Node("r", "relu", ("m",)), Node("m", "relu", ("x",))), "r")
        with self.assertRaisesRegex(ValueError, "forward reference"):
            graph.validate()

    def test_invalid_graphs(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Graph((Node("m", "relu", ("x",)), Node("m", "relu", ("x",))), "m").validate()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            Graph((Node("m", "exec", ("x",)),), "m").validate()


if __name__ == "__main__":
    unittest.main()
