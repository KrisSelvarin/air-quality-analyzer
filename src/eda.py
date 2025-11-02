# eda.py - for exploratory data analysis

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.logger_config import setup_logger

# Logger
logger = setup_logger(__name__)

class Exploratory:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def distribution(self,
                    title: str, 
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

        try: 
            # function theme and palette
            logger.info(f'Set function theme and palette to \'{style}\' and \'{palette}\'')
            sns.set_theme(style=style, palette=palette)

            # create matplotlib OOP
            logger.info('Created plot figure')
            fig, ax = plt.subplots(figsize=(8,4))

            # Safety check
            if var not in self.df.columns:
                raise KeyError(f"Columns {var} not found in DataFrame.")

            # Create histogram
            logger.info(f'Created histogram with variable {var}')
            sns.histplot(self.df[var], bins=50, kde=True, ax=ax)

            # Set title and label
            logger.info(f'Set title to {title}')
            ax.set_title(title, fontsize=15, fontweight='bold')
            logger.info(f'Set xlabel to {var if xlabel is None else xlabel}')
            ax.set_xlabel(var if xlabel is None else xlabel)

            # tight layout
            plt.tight_layout()

            # return figure
            return fig
        
        except Exception as e:
            logger.error(f'Error while generating distribution plot {title}: %s', e, exc_info=True)
            raise

    def trends(
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
        Trends of variable over time using line graph.

        Args:
            title (str):                Title of the plot/graph
            x (str):                    Variable/column of the dataframe to be plotted on x-axis.
            y (str):                    Variable/column of the dataframe to be plotted on y-axis.
            xlabel (str, optional):     x-axis label of the plot/graph. Defaults to x (column name)
            ylabel (str, optional):     x-axis label of the plot/graph. Defaults to y (column name)
            style (str, optional):      Set seaborn style. Defaults to darkgrid.
            palette (str, optional):    Set seaborn palette. Defaults to flare.

        Returns:
            matplotlib.figure.Figure:   The resulting Matplotlib figure object.

        """

        try: 
            # function theme and palette
            logger.info(f'Set function theme and palette to \'{style}\' and \'{palette}\'')
            sns.set_theme(style=style, palette=palette)

            # create matplotlib OOP
            logger.info('Created plot figure')
            fig, ax = plt.subplots(figsize=(10,6))

            # group by date and get daily averages
            df_line = (self.df.groupby(self.df.index.floor('D'))
                    .mean()
                    .round(2)
                    .reset_index()
                    .rename(columns={'DateTime': 'Date'})
                    )

            # safety check
            if x not in df_line.columns or y not in df_line.columns:
                raise KeyError(f"Columns {x} or {y} not found in DataFrame.")

            # create line plot
            logger.info(f'Created lineplot with x as {x} and y as {y}')
            sns.lineplot(df_line, x=df_line[x], y=df_line[y], ax=ax)
            ax.set_xlim(df_line[x].min(), df_line[x].max())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)

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
            logger.error(f'Error while generating trend plot {title}: %s', e, exc_info=True)
            raise

    def relation(
            self,
            title: str,
            *,
            x: str,
            y: str,
            xlabel: str = None,
            ylabel: str = None,
            style: str = 'darkgrid',
            palette: str = 'crest'
    ):
        """
        Relationship of two variables using scatter plot.
        
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
            # function theme and palette
            logger.info(f'Set function theme and palette to \'{style}\' and \'{palette}\'')
            sns.set_theme(style=style, palette=palette)

            # create matplotlib OOP
            logger.info('Created plot figure')
            fig, ax = plt.subplots(figsize=(10,6))

            # safety check
            if x not in self.df.columns or y not in self.df.columns:
                raise KeyError(f"Columns {x} or {y} not found in DataFrame.")
            
            # create scatter plot
            logger.info(f'Created scatter plot with x as {x} and y as {y}')
            sns.scatterplot(data=self.df, x=x, y=y, hue='RH', alpha=0.6, ax=ax)

            # create regplot
            logger.info(f'Created regplot')
            sns.regplot(data=self.df, x=x, y=y, scatter=False, color='red', line_kws={'lw': 1}, ax=ax)

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
            logger.error(f'Error while generating relation plot {title}: %s', e, exc_info=True)
            raise