import unittest

from rewards import correctness_reward, extract_final_number, normalize_number


class RewardFunctionTests(unittest.TestCase):
    def test_extract_hash_answer(self):
        self.assertEqual(extract_final_number("Reasoning... #### 42"), "42")

    def test_extract_last_number(self):
        self.assertEqual(extract_final_number("3 + 4 = 7. Final answer: 7"), "7")

    def test_normalize_equivalent_numbers(self):
        self.assertEqual(normalize_number("42.0"), "42")

    def test_correctness_reward(self):
        self.assertEqual(correctness_reward(["Final answer: 42"], ["#### 42"]), [1.0])
        self.assertEqual(correctness_reward(["Final answer: 41"], ["#### 42"]), [0.0])


if __name__ == "__main__":
    unittest.main()
