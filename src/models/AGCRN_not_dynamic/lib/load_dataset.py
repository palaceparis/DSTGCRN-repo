import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.seasonal import seasonal_decompose


def month_to_season(month):
    if month in [12, 1, 2]:
        return 0  # winter
    elif month in [3, 4, 5]:
        return 1  # spring
    elif month in [6, 7, 8]:
        return 2  # summer
    else:
        return 3  # fall


def load_and_transform_data(args):
    # Load and transform the "emissions" data
    emi_file = args.dataset.emissions
    emissions = pd.read_csv(emi_file)
    # Rename the first column
    emissions = emissions.rename(columns={"Unnamed: 0": "Date"})
    # Convert the 'Date' column to datetime
    emissions["Date"] = pd.to_datetime(emissions["Date"])
    # Set the 'Date' column as the index
    emissions = emissions.set_index("Date")

    # Decompose the log-transformed emissions data
    emissions_decomposed = emissions.apply(lambda x: seasonal_decompose(x))
    emissions_np = emissions.to_numpy()

    # Extract the trend, seasonal, and residual components
    emissions_trend = np.stack([x.trend for x in emissions_decomposed], axis=1)
    emissions_seasonal = np.stack([x.seasonal for x in emissions_decomposed], axis=1)
    emissions_residual = np.stack([x.resid for x in emissions_decomposed], axis=1)

    # Drop any missing values from the decomposition
    emissions_trend_df = pd.DataFrame(emissions_trend)
    emissions_seasonal_df = pd.DataFrame(emissions_seasonal)
    emissions_residual_df = pd.DataFrame(emissions_residual)

    # Record line numbers where NA values occur
    trend_na_lines = emissions_trend_df.isna().any(axis=1)
    seasonal_na_lines = emissions_seasonal_df.isna().any(axis=1)
    residual_na_lines = emissions_residual_df.isna().any(axis=1)

    # Use the line numbers to slice other arrays
    trend_lines_to_keep = ~trend_na_lines
    seasonal_lines_to_keep = ~seasonal_na_lines
    residual_lines_to_keep = ~residual_na_lines

    # Apply line numbers to emissions_np
    emissions_np = emissions_np[trend_lines_to_keep]

    emissions_trend = np.array(emissions_trend_df.loc[trend_lines_to_keep])
    emissions_seasonal = np.array(emissions_seasonal_df.loc[seasonal_lines_to_keep])
    emissions_residual = np.array(emissions_residual_df.loc[residual_lines_to_keep])

    # Apply line numbers to emissions_trend
    emissions_seasonal = emissions_seasonal[trend_lines_to_keep]

    if args.dataset.dataset == "CN":
        # Load and transform the "tmp" data
        tmp = pd.read_csv(args.tmp)
        constant = abs(tmp.min().min()) + 1e-6
        tmp_shifted = tmp + constant
        tmp_np = tmp_shifted.to_numpy()
        # Apply line numbers to tmp_np
        tmp_np = tmp_np[trend_lines_to_keep]

        # Load and transform the "aqi" data
        aqi = pd.read_csv(args.aqi)
        aqi_np = aqi.to_numpy()
        # Apply line numbers to aqi_np
        aqi_np = aqi_np[trend_lines_to_keep]

    # Date range
    start_date = datetime(2019, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(emissions.shape[0])]

    # Season
    season = np.array([month_to_season(date.month) for date in dates])

    # One-hot encoding for season
    encoder = OneHotEncoder(sparse=False)
    season_1hot = encoder.fit_transform(season.reshape(-1, 1))
    season_1hot = np.repeat(season_1hot[:, np.newaxis, :], emissions.shape[1], axis=1)
    # Apply line numbers to season_1hot
    season_1hot = season_1hot[trend_lines_to_keep]

    # Month
    months = np.array([date.month for date in dates])

    # One-hot encoding for months
    month_1hot = encoder.fit_transform(months.reshape(-1, 1))
    month_1hot = np.repeat(month_1hot[:, np.newaxis, :], emissions.shape[1], axis=1)
    # Apply line numbers to month_1hot
    month_1hot = month_1hot[trend_lines_to_keep]

    if args.dataset.dataset == "CN":
        data = np.concatenate(
            (
                emissions_np[:, :, np.newaxis],
                emissions_trend[:, :, np.newaxis],
                emissions_seasonal[:, :, np.newaxis],
                emissions_residual[:, :, np.newaxis],
                tmp_np[:, :, np.newaxis],
                aqi_np[:, :, np.newaxis],
                season_1hot,
                month_1hot,
            ),
            axis=2,
        )
    else:
        data = np.concatenate(
            (
                emissions_np[:, :, np.newaxis],
                emissions_trend[:, :, np.newaxis],
                emissions_seasonal[:, :, np.newaxis],
                emissions_residual[:, :, np.newaxis],
                # tmp_np[:, :, np.newaxis],
                # aqi_np[:, :, np.newaxis],
                season_1hot,
                month_1hot,
            ),
            axis=2,
        )

    # Remove the last entry so that val and test could be the same
    data = data[:-1]
    return data
