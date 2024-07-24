import unittest
from examples.run import run_selected


class TestExamples(unittest.TestCase):

    def test_selected_examples(self):
        success = run_selected(
            ['Example_01'])
        
        self.assertTrue(all(success))

if __name__ == '__main__':
    unittest.main()
