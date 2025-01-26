import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import os

def add_hour_quarter(df):
    df = df.with_columns(
        pl.col("index").str.slice(11, 2).cast(pl.Int32).alias("hour"),  # Extract hour (HH)
        pl.col("index").str.slice(14, 2).cast(pl.Int32).alias("minute")  # Extract minute (MM)
    )
    df = df.with_columns(
        pl.when(pl.col("minute") < 15)
        .then(1)
        .when(pl.col("minute") < 30)
        .then(2)
        .when(pl.col("minute") < 45)
        .then(3)
        .otherwise(4)
        .alias("group_minute")  # Create group_minute column
    )
    return df

def compute_response_function(df: pl.DataFrame, tau_max: int) -> np.ndarray:
    """
    Computes the response function R(tau) for all lags up to tau_max.

    Args:
        df (pl.DataFrame): DataFrame containing 'bid-price', 'ask-price', and 'trade-price'.
        tau_max (int): Maximum lag for which to compute the response function.

    Returns:
        np.ndarray: Array of response values for each tau from 1 to tau_max.
    """
    # Compute mid-price
    df = df.with_columns(
        ((pl.col("bid-price") + pl.col("ask-price")) / 2).alias("mid-price")
    )

    # Compute the sign s = sign(trade-price - mid-price), fill trade-price nulls with 0
    df = df.with_columns(
        (pl.col("trade-price").fill_null(0) - pl.col("mid-price")).sign().alias("s")
    )

    # Extract the mid-price and signal as NumPy arrays
    m = df["mid-price"].to_numpy()
    s = df["s"].to_numpy()

    # Ensure tau_max does not exceed data length
    tau_max = min(tau_max, len(m) - 1)

    # Compute response for each tau
    response = []
    for tau in range(1, tau_max + 1):
        # We only compute if we have enough data to shift
        if len(m[tau:]) > 0 and len(m[:-tau]) > 0:
            shifted_diff = m[tau:] - m[:-tau]  # m_{n+tau} - m_n
            response.append(np.mean(s[:-tau] * shifted_diff))
        else:
            response.append(np.nan)

    return np.array(response)

def compute_hourly_response(df: pl.DataFrame, tau_max: int) -> dict:
    """
    Computes the response function R(tau) for each hour in the dataset.

    Args:
        df (pl.DataFrame): DataFrame containing 'hour', 'bid-price', 'ask-price', and 'trade-price'.
        tau_max (int): Maximum lag for which to compute the response function.

    Returns:
        dict: A dictionary where keys are hourly timestamps and values are response arrays.
    """

    # Ensure 'hour' column exists
    if "hour" not in df.columns:
        raise ValueError("'hour' column is required in the DataFrame.")
    
    df = df.filter(pl.col("hour") != 16)

    # Group by 'hour' and compute the response function for each group
    hourly_responses = {}
    for hour, group in df.group_by("hour"):
        response = compute_response_function(group, tau_max)
        hourly_responses[hour] = response

    return hourly_responses

def filter_by_date(df: pl.DataFrame, target_date: str) -> pl.DataFrame:
    """
    Filters the Polars DataFrame by the given date string
    and returns a new DataFrame.
    """
    return df.filter(pl.col("date") == target_date)

def compute_minute_quarter_response(df: pl.DataFrame, tau_max: int) -> dict:
    """
    Computes the response function R(tau) for each 1-minute period, grouped by hour and quarter.

    Args:
        df (pl.DataFrame): DataFrame containing 'hour', 'group_minute', 'bid-price', 'ask-price', and 'trade-price'.
        tau_max (int): Maximum lag for which to compute the response function.

    Returns:
        dict: A dictionary where keys are tuples (hour, quarter) and values are response arrays.
    """

    # Ensure required columns exist
    if "hour" not in df.columns or "group_minute" not in df.columns:
        raise ValueError("'hour' and 'group_minute' columns are required in the DataFrame.")
    
    df = df.filter(pl.col("hour") != 16)

    # Group by 'hour' and 'group_minute' and compute the response function for each group
    responses = {}
    for (hour, quarter), group in df.group_by(["hour", "group_minute"]):
        response = compute_response_function(group, tau_max)
        responses[(hour, quarter)] = response

    return responses

def plot_hourly_responses(hourly_responses: dict, tau_max: int):
    """
    Plots the response functions for all hours.

    Args:
        hourly_responses (dict): A dictionary where keys are hours and values are response arrays.
        tau_max (int): Maximum lag (tau) for the x-axis.
    """
    plt.figure(figsize=(12, 8))

    for hour, response in hourly_responses.items():
        # Generate x-axis values matching the length of the response
        tau_values = range(1, len(response) + 1)
        plt.plot(
            tau_values, response, label=f'Hour {hour}'
        )

    plt.xlabel('Lag (tau)')
    plt.ylabel('Response Function R(tau)')
    plt.title('Response Function Comparison for every hour')
    plt.legend(title="Hour")
    plt.grid()
    plt.show()

def plot_quarterly_responses(hourly_responses: dict, tau_max: int):
    """
    Plots the response functions for all hours.

    Args:
        hourly_responses (dict): A dictionary where keys are hours and values are response arrays.
        tau_max (int): Maximum lag (tau) for the x-axis.
    """
    plt.figure(figsize=(12, 8))

    for hour, response in hourly_responses.items():
        # Generate x-axis values matching the length of the response
        tau_values = range(1, len(response) + 1)
        plt.plot(
            tau_values, response, label=f'(Hour, Quarter): {hour}'
        )

    plt.xlabel('Lag (tau)')
    plt.ylabel('Response Function R(tau)')
    plt.title('Response Function Comparison for every 15 minutes')
    plt.legend(title="(Hour, Quarter)")
    plt.grid()
    plt.show()

def plot_quarterly_responses_subplots(files, tau_max):
    """
    Plots the response functions for multiple files in subplots.

    Args:
        files (list): List of file names to process and plot.
        tau_max (int): Maximum lag (tau) for the x-axis.
    """
    # Calculate rows and columns for 7 subplots
    rows = 3
    cols = 3  # Set columns to 3 to fit in 3 rows

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    for idx, file in enumerate(files):
        if idx >= len(axes):
            break  # Stop if there are more files than subplots

        ax = axes[idx]
        stock = pl.read_parquet("processed/final_yearly/" + file)
        stock = add_hour_quarter(stock)
        stock = filter_by_date(stock, "2010-05-06")
        response_quarterly = compute_minute_quarter_response(stock, tau_max)

        # Plot responses for the current file
        for hour, response in response_quarterly.items():
            tau_values = range(1, len(response) + 1)
            ax.plot(tau_values, response, label=f'(Hour, Quarter): {hour}')

        # Add titles and labels
        ax.set_title(f"File: {file}")
        ax.set_xlabel("Lag (tau)")
        ax.set_ylabel("Response")
        ax.grid()

    # Hide unused subplots
    for ax in axes[len(files):]:
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def add_month(df):
    return df.with_columns(
        df["date"].str.slice(5, 2).alias("month")  # Extract characters 5–7 from the 'date' column
    )

def compute_impact_equal_portfolio(file_paths, tau_max, months):
    """
    Compute portfolio weights where each stock has the same impact.

    Args:
        file_paths (list): List of file paths for the 22 assets.
        tau_max (int): Maximum lag (tau) for the response function.
        month (str): Month to sample (default: "03" for March).

    Returns:
        dict: Asset weights for an impact-equal portfolio.
    """
    inverse_peaks = {}

    # Step 1: Compute total response peaks for each asset
    for file in file_paths:
        df = pl.read_parquet('processed/final_yearly/' + file)
        df = add_month(df)
        df = df.filter(df["month"].is_in(months))  # Filter for rows where month is in the list
        df = add_hour_quarter(df)
        df = df.drop_nulls()  # Remove rows with NaN values


        # Compute response function
        response_quarterly = compute_minute_quarter_response(df, tau_max)

        # Calculate total response peak for this asset
        total_peak = np.sum([np.median(np.abs(values)) for values in response_quarterly.values()])

        # Store the inverse of the total peak
        inverse_peaks[file] = 1 / total_peak if total_peak > 0 else 0

    # Step 2: Normalize weights to sum to 1
    total_inverse = sum(inverse_peaks.values())
    portfolio_weights = {asset: inverse / total_inverse for asset, inverse in inverse_peaks.items()}

    return portfolio_weights

def get_portfolio_response_peaks(file_paths, portfolio_weights, tau_max, months, threshold_percentile=70):
    """
    Get response function peaks distribution for the portfolio and identify high-peak dates and hours.

    Args:
        file_paths (list): List of file paths for the assets.
        portfolio_weights (dict): Portfolio weights for each asset.
        tau_max (int): Maximum lag (tau) for the response function.
        months (list): List of months to consider (e.g., ["03"] for March).
        threshold_percentile (int): Percentile to define "high" peaks (default: 95).

    Returns:
        dict: Dates and hours with high response function peaks for the portfolio.
    """
    # Initialize aggregated response function
    portfolio_response = {}

    for file in file_paths:

        # Load and preprocess data
        df = pl.read_parquet('processed/final_yearly/' + file)
        df = add_month(df)
        df = df.filter(df["month"].is_in(months))
        df = add_hour_quarter(df)
        df = df.drop_nulls()

        for date in df["date"].unique().sort():
            # Compute response function
            date_df = df.filter(df["date"] == date)  # Filter rows for the current date
            response_quarterly = compute_minute_quarter_response(date_df, tau_max)
            for (hour, quarter), values in response_quarterly.items():
                if len(values) == 0:  # Skip empty arrays
                    continue
                response_peak = np.median(np.abs(values))
                key = (date, hour, quarter)
                if key not in portfolio_response:
                    portfolio_response[key] = 0
                portfolio_response[key] += response_peak * portfolio_weights[file]

    # Determine threshold for high peaks
    all_peaks = np.array(list(portfolio_response.values()))
    high_peak_threshold = np.percentile(all_peaks, threshold_percentile)

    # Identify dates and hours with high peaks
    high_peaks = {
        key: value
        for key, value in portfolio_response.items()
        if value >= high_peak_threshold
    }

    return {"high_peaks": high_peaks, "threshold": high_peak_threshold}

def get_stocks_kept():
    tau_max = 1000
    stocks_kept = []  # List to store files where the max key is (14, 4) or (15, 1)
    stocks_not_kept = []

    for file in os.listdir("processed/final_yearly/"):
        stock = pl.read_parquet("processed/final_yearly/" + file)
        stock = add_hour_quarter(stock)
        stock = filter_by_date(stock, "2010-05-06")
        response_hourly = compute_hourly_response(stock, tau_max)
        response_quarterly = compute_minute_quarter_response(stock, tau_max)

        # Use a generator to find the key and max absolute value
        max_key, max_value = max(
            ((key, np.max(np.abs(values))) for key, values in response_quarterly.items()),
            key=lambda x: x[1],  # Compare based on the absolute maximum value
        )

        # Check if the max key is either (14, 4) or (15, 1)
        if max_key in [(14,3), (14, 4), (15, 1)]:
            stocks_kept.append(file)  # Add the file name to stocks_kept
        else:
            stocks_not_kept.append(file)
        
        return stocks_kept
    

def get_high_peak_dates_portfolio(file_paths, tau_max, months):
    tau_max = 1000
    portfolio_weights = compute_impact_equal_portfolio(file_paths, tau_max, months)
    results = get_portfolio_response_peaks(file_paths, portfolio_weights, tau_max, months)

    return results

