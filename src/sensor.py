# sensor.py - for sensor performance evaluation
# my first time touching machine learning 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger_config import setup_logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

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
        Visualizes and quantifies the linear relationship between two sensor-related 
        variables using a scatter plot and regression line. 

        This function helps assess how well a sensor’s measured data (x) aligns 
        with ground-truth or reference data (y) by displaying their correlation 
        and fitted regression trend. It also annotates the computed Pearson 
        correlation coefficient directly on the plot for quick interpretation.

        Args:
            title (str):  
                Title of the plot (e.g., "Sensor vs Ground Truth Correlation").
                
            x (str):  
                Name of the column in `self.df` to be plotted on the x-axis 
                (typically the sensor reading or input variable).
                
            y (str):  
                Name of the column in `self.df` to be plotted on the y-axis 
                (typically the reference or ground truth value).
                
            xlabel (str, optional):  
                Custom label for the x-axis.  
                If not provided, defaults to the column name specified by `x`.
                
            ylabel (str, optional):  
                Custom label for the y-axis.  
                If not provided, defaults to the column name specified by `y`.
                
            style (str, optional):  
                Seaborn visual style for the plot (e.g., "whitegrid", "darkgrid").  
                Defaults to `"darkgrid"`.
                
            palette (str, optional):  
                Seaborn color palette to use for styling the plot.  
                Defaults to `"flare"`.

        Returns:
            matplotlib.figure.Figure:  
                The resulting Matplotlib figure object containing the correlation plot 
                with scatter points, regression line, and annotated correlation coefficient.

        Raises:
            KeyError:
                If either the specified `x` or `y` column does not exist in `self.df`.
                
            Exception:
                For any unexpected error encountered during correlation computation 
                or plot generation.

        Notes:
            - The correlation coefficient (r) ranges from -1 to +1:
                * **r ≈ +1** → strong positive correlation  
                * **r ≈ 0** → no linear correlation  
                * **r ≈ -1** → strong negative correlation
            - This function is useful for evaluating how closely a sensor’s readings 
            match the true measurements before applying calibration models.
            - The plotted regression line provides a quick visual cue of the 
            relationship’s strength and direction.
        """
        
        try:
            # function for theme and palette
            logger.info(f'Set function theme and palette to \'{style}\' and \'{palette}\'')
            sns.set_theme(style=style, palette=palette)

            # Create matplotlib OOP
            logger.info('Creating plot figure')
            fig, ax = plt.subplots(figsize=(10,6))

            # safety check
            if x not in self.df.columns or y not in self.df.columns:
                raise KeyError(f"Columns {x} or {y} not found in DataFrame")

            # create scatter plot
            logger.info(f'Creating scatter plot with x as {x} and y as {y}.')
            sns.scatterplot(data=self.df, x=x, y=y, alpha=0.6, ax=ax)

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
            logger.error(f'Error while generating sensor correlation plot {title}: %s', e, exc_info=True)
            raise
        
    def residual_analysis(
            self,
            feature: str,
            target: str,
            title: str,
            xlabel: str = 'Prediction Error (Actual - Predicted)',
            *,
            style: str = 'darkgrid',
            palette: str = 'flare'
    ):
        """
        Perform residual analysis for a selected feature against a target variable.

        This method fits a simple linear regression model between the specified feature 
        (independent variable) and target (dependent variable), then analyzes and visualizes 
        the residuals — the differences between actual and predicted values. A well-calibrated 
        model should have residuals that are small, roughly normally distributed, and centered 
        around zero.

        Args:
            feature (str):  
                Name of the input feature column (independent variable). Must exist in `self.df`.
                
            target (str):  
                Name of the target column (dependent variable). Must exist in `self.df`.
                
            title (str):  
                Title of the residual plot (e.g., "Residuals: Sensor vs True CO").
                
            xlabel (str, optional):  
                Label for the x-axis of the plot. Defaults to 
                "Prediction Error (Actual - Predicted)".
                
            style (str, optional):  
                Seaborn visual style to apply. Defaults to "darkgrid".
                
            palette (str, optional):  
                Seaborn color palette to use. Defaults to "flare".

        Returns:
            tuple:
                fig (matplotlib.figure.Figure):  
                    The Matplotlib figure object containing the residual distribution plot.
                    
                model (sklearn.linear_model.LinearRegression):  
                    The trained LinearRegression model fitted on the given feature and target.
                    
                X (pandas.DataFrame):  
                    The feature data used to train and predict, containing the selected `feature`.
    
                y (pandas.Series):  
                    The actual target values corresponding to `X`.
                    
                y_pred (numpy.ndarray):  
                    The predicted target values generated by the regression model.

        Raises:
            KeyError:  
                If either `feature` or `target` is not found in `self.df`.
                
            Exception:  
                If any unexpected error occurs during model fitting or plotting.

        Notes:
            - Residuals close to zero indicate accurate predictions.
            - A symmetric, bell-shaped histogram centered at zero suggests unbiased calibration.
            - Use this function to evaluate sensor calibration quality or model fit performance.
            - The method logs key steps and errors for traceability.
        """

        try:
            # get X (matrix/dataframe) and y(series)
            X = self.df[[feature]]
            y = self.df.loc[X.index, target]

            # safety check
            if feature not in self.df.columns or target not in self.df.columns:
                raise KeyError(f"Columns {feature} or {target} not found in DataFrame")

            # create regression model
            logger.info('Creating regression model.')
            model = LinearRegression().fit(X, y)
            logger.info(f'Model Created. Coefficient = {model.coef_} Intercept = {model.intercept_}')

            logger.info('Predicting values using generated model')
            y_pred = model.predict(X)
            residuals = y - y_pred

            # setting the style and palette for seaborn
            logger.info(f'Set function theme and palette to \'{style}\' and \'{palette}\'')
            sns.set_theme(style=style, palette=palette)

            # plotting the residuals
            logger.info('Creating plot figure')
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(x=residuals, kde=True, ax=ax)

            # labels
            logger.info(f'Set title to {title}')
            ax.set_title(title, fontsize=20, fontweight='bold')
            logger.info(f'Set xlabel to {xlabel}')
            ax.set_xlabel(xlabel)

            # tight lay-out
            plt.tight_layout()

            # return fig, model, residuals
            return fig, model, X, y, y_pred
        
        except Exception as e:
            logger.error(f'Error while generating residuals histplot {title}: %s', e, exc_info=True)
            raise

    @staticmethod
    def accurate(
            y,
            y_pred,
    ):
        """
        Compute accuracy metrics to quantify sensor prediction performance.

        This method evaluates how closely the predicted values match the actual 
        ground truth measurements using two common error magnitude metrics:
        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)

        Args:
            y (array-like or pandas.Series):  
                Actual (true) target values.
                
            y_pred (array-like or pandas.Series):  
                Predicted target values generated by a model or sensor.

        Returns:
            tuple:
                mae (float):  
                    Mean Absolute Error — the average absolute difference 
                    between predicted and actual values.
                    
                rmse (float):  
                    Root Mean Squared Error — the square root of the average 
                    squared differences between predicted and actual values.

        Raises:
            ValueError:  
                If the input arrays have different lengths or contain invalid values.
                
            Exception:  
                For any unexpected error during metric computation.

        Notes:
            - Lower MAE and RMSE values indicate higher prediction accuracy.
            - MAE is more robust to outliers, while RMSE penalizes larger errors more strongly.
            - All computations and results are logged for traceability.
        """

        try:
            # Mean Absolute Error
            logger.info('Calculating Mean Absolute Error')
            mae = mean_absolute_error(y, y_pred)
            logger.info(f'Mean Absolute Error: {mae}')

            # Root Mean Squared Error
            logger.info('Calculating Root Mean Squared Error')
            rmse = root_mean_squared_error(y, y_pred)
            logger.info(f'Root Mean Squared Error: {rmse}')

            return mae, rmse
        
        except ValueError as ve:
            logger.error('ValueError during accuracy metric computation: %s', ve, exc_info=True)
            raise

        except Exception as e:
            logger.error('Unexpected error during accuracy metric computation: %s', e, exc_info=True)
            raise

    @staticmethod
    def split_validation(X, y):
        """
        Evaluate model generalization using train-test split and R² metrics.

        This method splits the input dataset into training and testing subsets,
        fits a linear regression model, and evaluates its performance using
        the coefficient of determination (R²) on both sets. The results
        indicate how well the model generalizes to unseen data.

        Args:
            X (array-like or pandas.DataFrame):  
                Feature matrix containing predictor variables.

            y (array-like or pandas.Series):  
                Target variable corresponding to X.

        Returns:
            tuple:
                r2_train (float):  
                    Coefficient of Determination (R²) on the training dataset, 
                    indicating how well the model fits the training data.

                r2_test (float):  
                    Coefficient of Determination (R²) on the testing dataset, 
                    reflecting how well the model generalizes to unseen data.

        Raises:
            ValueError:  
                If the input data shapes are incompatible or contain invalid values.

            Exception:  
                For any unexpected error encountered during data splitting, 
                model training, or metric computation.

        Notes:
            - A higher R² value indicates better model performance.  
            - A large difference between training and testing R² may suggest overfitting.  
            - All operations and results are logged for traceability and debugging.
        """

        try:
            # splitting the data
            logger.info('Splitting the data for training and testing')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # model
            logger.info('Creating model from training data')
            model = LinearRegression().fit(X_train, y_train)
            logger.info('Model Created.')

            # Coefficient of Determination (R²)
            logger.info('Calculating R² of training data')
            r2_train = model.score(X_train, y_train)
            logger.info(f'R² (Training data): {r2_train}')

            logger.info('Calculating R² of test data')
            r2_test = model.score(X_test, y_test)
            logger.info(f'R² (Test data): {r2_test}')

            # return
            return r2_train, r2_test

        except ValueError as ve:
            logger.error('ValueError during split validation: %s', ve, exc_info=True)
            raise

        except Exception as e:
            logger.error('Unexpected error during split validation: %s', e, exc_info=True)
            raise