import unittest
from core.datasets.utils import (
    LEN_EXTRA_WORDS,
    BOS_I,
    EOS_I,
    create_tf_input,
    create_tf_target,
)


class MyTestCase(unittest.TestCase):
    def test_tf_target_creation(self):
        x = [1, 2, 3]
        result = [i + LEN_EXTRA_WORDS for i in x] + [EOS_I]

        self.assertListEqual(create_tf_target(x), result)

    def test_tf_input_creation(self):
        x = [1, 2, 3]
        result = [BOS_I] + [i + LEN_EXTRA_WORDS for i in x]

        self.assertListEqual(create_tf_input(x), result)


if __name__ == "__main__":
    unittest.main()
