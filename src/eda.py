# eda.py - for exploratory data analysis

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from src.logger_config import setup_logger

# Logger
logger = setup_logger(__name__)

class Exploratory:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def distribution(self, title: str, 
                    var: str, 
                    xlabel: str = None, *, 
                    style: str = 'darkgrid', 
                    palette: str = 'crest'
                ):
        """
        Distribution and behaviour of variable using histogram.

        Args:
            title (str):                Title of the plot/graph
            var (str):                  Variable/column of the dataframe to be plotted.
            xlabel (str, optional):     x-axis label of the plot/graph. Defaults to var.
            style (str, optional):      Set seaborn style. Defaults to darkgrid.
            palette (str, optional):    Set seaborn palette. Defaults to crest.

        Returns:
            matplotlib.figure.Figure:   The resulting Matplotlib figure object.
        """

        # function theme and palette
        logger.info(f'Set function theme and palette to \'{style}\' and \'{palette}\'')
        sns.set_theme(style=style, palette=palette)

        # create matplotlib OOP
        logger.info('Created plot figure')
        fig, ax = plt.subplots(figsize=(8,4))

        # Create histogram
        logger.info(f'Created histogram with variable {var}')
        sns.histplot(self.df[var], bins=50, kde=True, ax=ax)

        # Set title and label
        logger.info(f'Set title to {title}')
        ax.set_title(title, fontsize=15, fontweight='bold')
        logger.info(f'Set xlabel to {xlabel}')
        ax.set_xlabel(var if xlabel is None else xlabel)

        # tight layout and show
        plt.tight_layout()
        logger.info('Showing Figure.')
        plt.show()
        logger.info('Figure Shown.')

        # return figure
        return fig