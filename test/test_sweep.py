import unittest
from heat_battery.simulations.sweep import ParameterGrid, ParameterList, ParameterEvaluation, NoNumericalEffect

class TestSweep(unittest.TestCase):
    def setUp(self) -> None:

        def foo(value, value2):
            return value + value2

        self.pg = ParameterGrid(dict(    
            arg_1 = 1.0,
            arg_2 = ParameterList([1.0, 2.0]),
            arg_3 = ParameterEvaluation("SELF['arg_1'] + SELF['arg_2']", eval_as_code=True),
            arg_4 = ParameterEvaluation("SELF['arg_1'] + SELF['arg_2']", eval_as_code=False),
            arg_5 = ParameterEvaluation("{SELF['arg_1']} + {SELF['arg_2']}", eval_as_code=False),
            arg_6 = ParameterEvaluation("{SELF['PRIORITY']}/{SELF['SIGNATURE']}", eval_as_code=False),
            arg_7 = ParameterEvaluation("foo(SELF['arg_1'], SELF['arg_2'])", eval_as_code=True, defered_scope=dict(foo=foo)),
            arg_8 = NoNumericalEffect('no-effect')
        ))
        self.pg.instantiate()
    
    def test_parameter_evaluation(self):
        gen = self.pg.kde_parameters()

        results = list(gen)

        r0 = results[0]
        e0 = [
            1.0, 1.0, 2.0, "SELF['arg_1'] + SELF['arg_2']", '1.0 + 1.0', 
            '0/3e3446d231cbc270772c6c5c99a6f3d6f8dbbacbf2501251587c0e636850eb35', 
            2.0, 'no-effect',
        ]
        for i, e in enumerate(e0):
            self.assertEqual(r0[f'arg_{i+1}'], e)
        
        r1 = results[1]
        e1 = [
            1.0, 2.0, 3.0, "SELF['arg_1'] + SELF['arg_2']", '1.0 + 2.0', 
            '1/31121b6d60726f3a46cb32277b9980ec88e73f013f6d5f70735d422d5b33ce13', 
            3.0, 'no-effect',
        ]
        for i, e in enumerate(e1):
            self.assertEqual(r1[f'arg_{i+1}'], e)

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()