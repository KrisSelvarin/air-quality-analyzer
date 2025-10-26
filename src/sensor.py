# sensor.py - for sensor performance evaluation
# my first time touching machine learning 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger_config import setup_logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Logger
logger = setup_logger(__name__)

class Sensor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def sensor_correlation(
            self,
            title: str,
            *,
            x: str,
            y: str,
            xlabel: str = None,
            ylabel: str = None,
            style: str = 'darkgrid',
            palette: str = 'flare'

    ):
        """
        Check how sensor data correlates with data ground truth data.

        Args:
            title (str):                Title of the plot/graph
            x (str):                    Variable/column of the dataframe to be plotted on x-axis.
            y (str):                    Variable/column of the dataframe to be plotted on y-axis.
            xlabel (str, optional):     x-axis label of the plot/graph. Defaults to x (column name)
            ylabel (str, optional):     x-axis label of the plot/graph. Defaults to y (column name)
            style (str, optional):      Set seaborn style. Defaults to darkgrid.
            palette (str, optional):    Set seaborn palette. Defaults to crest.

        Returns:
            matplotlib.figure.Figure:   The resulting Matplotlib figure object.
        """
        
        try:
            # function for theme and palette
            logger.info(f'Set function theme and palette to \'{style}\' and \'{palette}\'')
            sns.set_theme(style=style, palette=palette)

            # Create matplotlib OOP
            logger.info('Created plot figure')
            fig, ax = plt.subplots(figsize=(10,6))

            # safety check
            if x not in self.df.columns or y not in self.df.columns:
                raise KeyError(f"Columns {x} or {y} not found in DataFrame")

            # create scatter plot
            logger.info(f'Creating scatter plot with x as {x} and y as {y}.')
            sns.scatterplot(data=self.data, x=x, y=y, alpha=0.6, ax=ax)

            # create regplot
            logger.info(f'Creating regplot.')
            sns.regplot(data=self.df, x=x, y=y, scatter=False, color='red', line_kws={'lw':1})

            # add computed correlation
            logger.info('Adding computed correlation value on plot')
            corr = self.df[[x, y]].corr().iloc[0, 1]
            ax.text(
                0.05, 0.95,                                 # position
                f'Correlation (r) = {corr:.3f}',            # text content
                transform=ax.transAxes,                     # relative to this Axes
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                va='top'                                    # top edge vertical alignment
            )

            # set title and label
            logger.info(f'Set title to {title}')
            ax.set_title(title, fontsize=20, fontweight='bold')
            logger.info(f'Set xlabel to {x if xlabel is None else xlabel}')
            ax.set_xlabel(x if xlabel is None else xlabel)
            logger.info(f'Set ylabel to {y if ylabel is None else ylabel}')
            ax.set_ylabel(y if ylabel is None else ylabel)

            # tight lay-out
            plt.tight_layout()

            # return fig
            return fig

        except Exception as e:
            logger.error(f'Error while generating sensor relation plot {title}: %s', e, exc_info=True)
            raise
        
    def residual_analysis(self):
        """
        Checks how far predictions deviate from actual ground truth. 
        If the residuals are small and centered around zero, calibration is good.
        (Error Distribution)
        """

        # TODO: use regression and use histogram (histplot) for residual distribution

    def accurate(self):
        """
        Quantifies how accurate the sensor predictions are.
        (Error Magnitude Metrics)
        """

        # TODO: use mean absolute error (MAE) and root mean squared error (RMSE)
        # MAE = average difference between true and predicted
        # RMSE = same as MAE but treat errors more harshly
        # Lower = better calibration

    def split_validation(self):
        """
        Evaluate how well a model generalizes.
        How well it performs on unseen data.
        """

        # TODO: test calibration on unseen data
        # use sklearn.model_selection -  train_test_split
        # if R² similar on train and test, then it's a stable calibration
        # if R² drops heavily on test, then it's overfitting/unreliable calibration
