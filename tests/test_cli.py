import contextlib
import io
import json
import unittest
from modelstack_lab.__main__ import main


class CLITests(unittest.TestCase):
    def run_cli(self, args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_doctor(self):
        code, content, error = self.run_cli(["doctor"])
        self.assertEqual((code, error), (0, ""))
        self.assertIn("python", json.loads(content))

    def test_graph(self):
        code, content, error = self.run_cli(["graph", "--size", "4", "--iterations", "2"])
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(content)["fused"]["operation_nodes"], 1)

    def test_fail_closed(self):
        code, content, error = self.run_cli(["graph", "--size", "200"])
        self.assertEqual(code, 2)
        self.assertEqual(content, "")
        self.assertIn("size", error)


if __name__ == "__main__":
    unittest.main()
