import unittest
import numpy as np
import webbrowser
import plotly.graph_objects as go
from heat_battery.data import Visualizer, pages
import time

class TestVizualizer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_basic(self):

        def create_single_plot():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(10), y=np.random.rand(10)))
            return [fig]
        
        def create_multiple_plots(n=1):
            figs = []
            for i in range(n):
                figs.append(create_single_plot()[0])
            return figs

        V = Visualizer()
        V.register_page(pages.FigurePage("Single", create_single_plot))
        V.register_page(pages.FigurePage("Multiple", create_multiple_plots, n=4))
        V.build_app()
        V.start_app()
        V.update_data()
        #webbrowser.open('http://127.0.0.1:8050/')
        
        for i in range(15):
            V.update_data()
            time.sleep(1)

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()