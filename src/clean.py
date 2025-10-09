# clean.py - data cleaning
import pandas as pd
from pathlib import Path
from src.logger_config import setup_logger

# logging
logger = setup_logger(__name__)

# directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / 'data' / 'raw' / 'AirQualityUCI.csv'
CLEANED_FILE = BASE_DIR / 'data' / 'cleaned' / 'AirQuality_cleaned.csv'

def data_clean() -> pd.DataFrame:
    """Handles data cleaning"""

    try:
        logger.info('Started data cleaning process')
        filename = RAW_FILE
        df = pd.read_csv(
            filename,
            delimiter=';',
            decimal=',',
        )

        # parsing and dropping columns
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
        df.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], inplace=True)
        logger.info('Parsed DateTime and dropped excess columns')

        # missing values
        df.dropna(how='all', inplace=True)
        logger.info('Dropped rows with NaN in all columns')

        # replace invalid values
        df.replace(-200, pd.NA, inplace=True)
        col = [col for col in df.columns if col not in ['DateTime']]
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')
        df.interpolate(method='linear', inplace=True)
        logger.info('Replaced invalid values with interploated values')

        # round numeric column
        c = [c for c in df.columns if c not in ['DateTime', 'AH']]
        df[c] = df[c].apply(lambda x: x.round(1))
        logger.info('Rounded off numeric columns')

        # index datetime
        df.set_index('DateTime', inplace=True)
        logger.info('Set DateTime as index')

        # save cleaned df to csv
        df.to_csv(CLEANED_FILE)
        logger.info('Saved cleaned data')

        # return DataFrame
        return df

    except Exception as e:
        logger.exception('Error during data cleaning')
        raise