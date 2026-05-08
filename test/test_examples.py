import unittest
from examples.run import run_selected


class TestExamples(unittest.TestCase):

    def test_example_01(self):
        run_selected(['Example_01'])
        
    # def test_example_02(self):
    #     run_selected(['Example_02'])
        
    # def test_example_03(self):
    #     run_selected(['Example_03'])
        
    # def test_example_04(self):
    #     run_selected(['Example_04'])

if __name__ == '__main__':
    unittest.main()
