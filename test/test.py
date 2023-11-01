import unittest

class TestDiscoveryOfTest(unittest.TestCase):
    def test_f(self):
        self.assertTrue('FOO'.isupper())

if __name__ == '__main__':
    unittest.main()