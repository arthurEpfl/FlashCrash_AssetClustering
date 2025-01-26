import pandas as pd
import numpy as np
import polars as pl
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple
import os
import itertools
from numpy import linalg as LA
from numpy.linalg import eig
from intervaltree import Interval, IntervalTree
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import community
from matplotlib.patches import Patch
import argparse
import pickle
import json

from clustering_script import *

# Asset names
asset_names = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DOW', 'HD', 'IBM',
    'INTC', 'JPM', 'KO', 'MMM', 'MRK', 'PG', 'TRV', 'UTX', 'V', 'VZ',
    'WMT', 'XOM'
]

# Step 1: Define the mapping from assets to domains
asset_to_domain = {
    'AAPL': 'Technology', 'IBM': 'Technology', 'INTC': 'Technology', 'MSFT': 'Technology',
    'AMGN': 'Healthcare', 'JNJ': 'Healthcare', 'MMM': 'Healthcare', 'MRK': 'Healthcare',
    'PFE': 'Healthcare', 'UNH': 'Healthcare',
    'AXP': 'Finance', 'GS': 'Finance', 'JPM': 'Finance', 'TRV': 'Finance',
    'BA': 'Industrials', 'CAT': 'Industrials', 'DOW': 'Industrials', 'RTX': 'Industrials',
    'UTX': 'Industrials',
    'CSCO': 'Telecommunications', 'VZ': 'Telecommunications',
    'CVX': 'Energy', 'XOM': 'Energy',
    'HD': 'Retail',
    'KO': 'Consumer Staples', 'WBA': 'Consumer Staples',
    'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
    'PG': 'Consumer Discretionary', 'V': 'Consumer Discretionary',
    'WMT': 'Consumer Discretionary'
}

domain_colors = {
    'Technology': '#1f77b4',
    'Healthcare': '#ff7f0e',
    'Finance': '#2ca02c',
    'Industrials': '#d62728',
    'Telecommunications': '#9467bd',
    'Energy': '#8c564b',
    'Retail': '#e377c2',
    'Consumer Staples': '#7f7f7f',
    'Consumer Discretionary': '#bcbd22'
}

def map_dates_to_strings(input_dict):
    """
    Maps dates and time data from the input dictionary to a specific string format.

    Args:
        input_dict (dict): A dictionary where each value is a tuple of the form 
                           (date tuple, np.float64 value), and the date tuple is 
                           ('YYYY-MM-DD', hour, quarter).

    Returns:
        list: A list of formatted strings in the format 'YYYY-MM-DD-HH:MM'.
    """
    quarter_to_minutes = {1: "00", 2: "15", 3: "30", 4: "45"}  # Quarter mapping
    formatted_strings = []

    for key, value in input_dict.items():
        date, hour, quarter = value[0]  # Extract the date tuple
        minutes = quarter_to_minutes.get(quarter, "00")  # Get the minutes for the quarter
        formatted_string = f"{date}-{hour:02}:{minutes}"  # Format the string
        formatted_strings.append(formatted_string)

    return formatted_strings

def main():
    # Define the periods and their corresponding file paths
    parser = argparse.ArgumentParser(description='Perform clustering and plot domains.')
    parser.add_argument('--demo', action='store_true', help='Use demo data paths.')
    args = parser.parse_args()
    demo = args.demo

    demo_periods_files = {
        'before': 'output_cov_matrices/before.parquet',
        'during': 'output_cov_matrices/during.parquet',
        'after': 'output_cov_matrices/after.parquet'
    }

    periods_files = {
        'before': 'output_cov_matrices3/before.parquet',
        'during': 'output_cov_matrices3/during.parquet',
        'after': 'output_cov_matrices3/after.parquet'
    }


    # Select the appropriate file paths based on the demo flag
    selected_files = demo_periods_files if demo else periods_files

    # Create directories for results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Create subdirectories for plots and dictionaries
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    dicts_dir = os.path.join(results_dir, 'dictionaries')
    os.makedirs(dicts_dir, exist_ok=True)

    # To store dictionaries for each period
    all_clusters_dict = {period: {} for period in selected_files.keys()}

    # Iterate over each period and its corresponding file
    for period, file_path in selected_files.items():
        print(f'\nProcessing Period: {period.capitalize()} | File: {file_path}')

        # Create a subdirectory for plots of the current period
        period_plots_dir = os.path.join(plots_dir, period)
        os.makedirs(period_plots_dir, exist_ok=True)

        # Load all covariance matrices for the period
        try:
            cov_matrices_dict = load_and_reshape_covariance(file_path)
        except Exception as e:
            print(f'Error loading file {file_path}: {e}')
            continue

        # Iterate over each date and its covariance matrix
        for date, cov_matrix in cov_matrices_dict.items():
            print(f'  Processing Date: {date}')

            try:
                # Step 1: Apply eigenvalue clipping
                clipped_cov_matrix = eigenvalue_clipping(cov_matrix)
                # Step 2: Fill diagonal with zeros
                np.fill_diagonal(clipped_cov_matrix, 0)
                clipped_cov_matrix = np.asarray(clipped_cov_matrix)
                # Step 3: Normalize the covariance matrix
                scaler = MinMaxScaler()
                adj_matrix = scaler.fit_transform(clipped_cov_matrix)
                # Step 4: Perform Louvain clustering
                partition_optimal = perform_louvain_clustering(adj_matrix)

                clusters_optimal = {}
                for node, cluster_id in partition_optimal.items():
                    # Ensure node index is within the range of asset_names
                    if node < len(asset_names):
                        asset_name = asset_names[node]
                    else:
                        asset_name = f'Asset_{node}'  # Placeholder for undefined assets
                    clusters_optimal.setdefault(cluster_id, []).append(asset_name)

                # Save the clusters in the dictionary for this period
                all_clusters_dict[period][date] = clusters_optimal

                # Step 6: Process the clusters to count domains
                cluster_domains = {}
                for cluster_id, assets in clusters_optimal.items():
                    # Initialize domain count for the cluster
                    domain_count = {domain: 0 for domain in domain_colors.keys()}
                    for asset in assets:
                        domain = asset_to_domain.get(asset, 'Unknown')
                        domain_count[domain] += 1
                    cluster_domains[cluster_id] = domain_count

                # Step 7: Define the save path for the plot
                save_filename = f'{period}_{date}.png'
                save_path = os.path.join(period_plots_dir, save_filename)

                # Step 8: Plot and save the cluster domains
                plot_cluster_domains(
                    cluster_domains=cluster_domains,
                    clusters_optimal=clusters_optimal,
                    domain_colors=domain_colors,
                    period=period,
                    date=date,
                    save_path=save_path
                )

                print(f'    Plot saved to: {save_path}')

            except Exception as e:
                print(f'    Error processing date {date}: {e}')
                continue

    # Save dictionaries for each period as separate files
    for period, clusters in all_clusters_dict.items():
        dict_file_path = os.path.join(dicts_dir, f'{period}_clusters.pkl')
        with open(dict_file_path, 'wb') as f:
            pickle.dump(clusters, f)
        print(f'Dictionary saved for period "{period}" at {dict_file_path}')

    print('\nAll clustering and plotting completed.')

if __name__ == '__main__':
    main()


def filter_by_date(df: pl.DataFrame, target_date: str) -> pl.DataFrame:
    """
    Filters the Polars DataFrame by the given date string
    and returns a new DataFrame.
    """
    return df.filter(pl.col("date") == target_date)

def hayashi_yoshida_covariance(dataframe, default_columns=("mid_price_return", "mid_price_return_right")):
    """
    Calculate the Hayashi-Yoshida covariance estimator for two numeric columns in a dataframe.

    Parameters:
        dataframe (pl.DataFrame): The input dataframe.
        default_columns (tuple): A tuple specifying default column names for the calculation.

    Returns:
        dict: A dictionary containing the Hayashi-Yoshida covariance, variances, and correlation.
    """
    if all(col in dataframe.columns for col in default_columns):
        col_x, col_y = default_columns
    else:
        raise ValueError(
            f"The dataframe must contain the specified columns: {default_columns}"
        )

    # Convert Polars DataFrame to Pandas for interval handling
    df = dataframe.to_pandas()

    # Extract necessary columns and drop NaNs
    df_x = df[["truncated_index", col_x]].dropna().sort_values(by="truncated_index").drop_duplicates(subset="truncated_index")
    df_y = df[["truncated_index_right", col_y]].dropna().sort_values(by="truncated_index_right").drop_duplicates(subset="truncated_index_right")

    # Parse truncated_index into proper datetime with millisecond precision
    df_x["truncated_index"] = pd.to_datetime(df_x["truncated_index"], format="%H:%M:%S.%f")
    df_y["truncated_index_right"] = pd.to_datetime(df_y["truncated_index_right"], format="%H:%M:%S.%f")

    # Create intervals for both series
    intervals_x = pd.IntervalIndex.from_arrays(
        df_x["truncated_index"][:-1], df_x["truncated_index"][1:], closed="right"
    )
    intervals_y = pd.IntervalIndex.from_arrays(
        df_y["truncated_index_right"][:-1], df_y["truncated_index_right"][1:], closed="right"
    )

    # Build an interval tree for Y
    tree_y = IntervalTree(
        Interval(begin.value, end.value, idx)
        for idx, (begin, end) in enumerate(zip(df_y["truncated_index_right"][:-1], df_y["truncated_index_right"][1:]))
    )

    # Calculate covariance
    # Calculate average of the product of returns
    covariance = 0
    #count = 0  # To keep track of the number of products
    for i, interval_x in enumerate(intervals_x):
        overlaps = tree_y.overlap(interval_x.left.value, interval_x.right.value)  # Use overlap instead of search
        for overlap in overlaps:
            j = overlap.data  # Index in df_y
            covariance += df_x[col_x].iloc[i] * df_y[col_y].iloc[j]
            #count += 1

    # Calculate average
    #average_product = covariance / count if count > 0 else 0

        
    # Calculate variances
    variance_x = np.sum(df_x[col_x] ** 2)
    variance_y = np.sum(df_y[col_y] ** 2)

    # Calculate correlation
    correlation = covariance / (np.sqrt(variance_x * variance_y) +1e-8)

    return {
        "hayashi_yoshida_covariance": covariance,
        "variance_x": variance_x,
        "variance_y": variance_y,
        "correlation": correlation,
    }

def split_dataframe_by_quarter(df: pl.DataFrame, date: str, start_time_str: str) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    clean_df = df.filter(
        (pl.col('mid_price_return').is_not_null()) & 
        (pl.col('mid_price_return_right').is_not_null())
    )
    
    start_datetime_str = f"{date} {start_time_str}:00"
    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
    
    if date != "2010-05-06-14:45" :
        quarter_duration = timedelta(minutes=15)
        start_time = start_datetime
        end_datetime = start_datetime + quarter_duration
    else :
        quarter_duration = timedelta(minutes=15)
        start_time = start_datetime - quarter_duration
        end_datetime = start_datetime + quarter_duration 
        end_datetime = end_datetime + quarter_duration
    
    start_quarter = start_time.strftime("%H:%M:%S")
    end_quarter = end_datetime.strftime("%H:%M:%S")
    
    df_before_quarter = clean_df.filter(pl.col("truncated_index") < start_quarter)
    df_during_quarter = clean_df.filter(
        (pl.col("truncated_index") >= start_quarter) & 
        (pl.col("truncated_index") <= end_quarter)
    )
    df_after_quarter = clean_df.filter(pl.col("truncated_index") > end_quarter)
    
    return df_before_quarter, df_during_quarter, df_after_quarter

def process_pair(asset1_file, asset2_file, date, start_time):
    """
    Process two parquet files to compute the Hayashi-Yoshida covariance, variances, and correlations.

    Parameters:
        asset1_file (str): Path to the first asset's parquet file.
        asset2_file (str): Path to the second asset's parquet file.
        date (str): The date to filter on (e.g., "2010-05-06").
        start_time (str): The start time of the quarter (e.g., "09:30").

    Returns:
        dict: A dictionary containing the results for each period (before, during, after).
    """

    # Load the parquet files
    asset1 = pl.read_parquet(asset1_file)
    asset2 = pl.read_parquet(asset2_file)

    # Filter by date
    asset1 = filter_by_date(asset1, date)
    asset2 = filter_by_date(asset2, date)


    # Add truncated index
    asset1 = asset1.with_columns(pl.col('index').str.slice(11, 12).alias('truncated_index'))
    asset2 = asset2.with_columns(pl.col('index').str.slice(11, 12).alias('truncated_index'))

    # Compute mid-price and mid-price returns
    asset1 = asset1.with_columns(((asset1['bid-price'] + asset1['ask-price']) / 2).alias('mid_price'))
    asset1 = asset1.with_columns(asset1['mid_price'].pct_change().alias('mid_price_return'))
    asset1 = asset1.filter(asset1['mid_price_return'] != 0).drop_nulls()

    asset2 = asset2.with_columns(((asset2['bid-price'] + asset2['ask-price']) / 2).alias('mid_price'))
    asset2 = asset2.with_columns(asset2['mid_price'].pct_change().alias('mid_price_return'))
    asset2 = asset2.filter(asset2['mid_price_return'] != 0).drop_nulls()

    # Remove rows where returns are zero or NaN
    asset1 = asset1.filter(asset1['mid_price_return'] != 0).drop_nulls()
    asset2 = asset2.filter(asset2['mid_price_return'] != 0).drop_nulls()


    # Join the dataframes
    result_df = asset1.join(asset2, on='truncated_index', how='full').sort('truncated_index')

    # Forward fill
    result_df = result_df.with_columns([
        pl.col('mid_price').fill_null(strategy='forward'),
        pl.col('mid_price_right').fill_null(strategy='forward')
    ])

    # Split the dataframe into periods
    df_before, df_during, df_after = split_dataframe_by_quarter(result_df, date, start_time)
    # Compute covariance and variances for each period
    results = {
        "before": hayashi_yoshida_covariance(df_before),
        "during": hayashi_yoshida_covariance(df_during),
        "after": hayashi_yoshida_covariance(df_after),
    }
    
    return results

def compute_covariance_matrices(data_folder, dates, output_folder):
    """
    Compute and append covariance matrices for all asset pairs across multiple dates.

    Parameters:
        data_folder (str): Path to the folder containing asset parquet files.
        dates (list[str]): List of dates and times to process (e.g., ["2010-05-06-14:45"]).
        output_folder (str): Path to the folder to save covariance matrices as parquet files.

    Returns:
        None
    """

    stocks_kept = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CSCO', 
    'CVX', 'DOW', 'HD', 'IBM', 'INTC', 'JPM', 
    'KO', 'MMM', 'MRK', 'PG', 'TRV', 'UTX', 
    'V', 'VZ', 'WMT', 'XOM'
    ]

    # List all parquet files in the data folder
    asset_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".parquet")]
    # Extract asset names from filenames
    asset_names = [os.path.basename(f).split('.')[0] for f in asset_files]

    asset_names = [stock for stock in asset_names if stock in stocks_kept]

    n_assets = len(asset_names)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize or read existing parquet files for each period
    def initialize_or_read_parquet(file_path, asset_names):
        if os.path.exists(file_path):
            return pl.read_parquet(file_path)
        else:
            schema = {"date": pl.Utf8}
            schema = {**schema, **{f"{name}_{name2}": pl.Float64 for name in asset_names for name2 in asset_names}}
            return pl.DataFrame(schema=schema)

    before_path = os.path.join(output_folder, "before.parquet")
    during_path = os.path.join(output_folder, "during.parquet")
    after_path = os.path.join(output_folder, "after.parquet")

    before_df = initialize_or_read_parquet(before_path, asset_names)
    during_df = initialize_or_read_parquet(during_path, asset_names)
    after_df = initialize_or_read_parquet(after_path, asset_names)

    print(dates)
    # Process each date
    for datestring in dates:
        try:
            date, start_time = datestring.rsplit("-", 1)
        except ValueError:
            print(f"Invalid date format: {datestring}. Expected format: YYYY-MM-DD-HH:MM")
            continue

        print(f"Processing date: {date} with start time: {start_time}")

        # Initialize covariance matrices for this date
        cov_matrix_before = np.zeros((n_assets, n_assets))
        cov_matrix_during = np.zeros((n_assets, n_assets))
        cov_matrix_after = np.zeros((n_assets, n_assets))

        # Iterate over all asset pairs
        for i, j in itertools.combinations_with_replacement(range(n_assets), 2):
            asset1_file = asset_files[i]
            asset2_file = asset_files[j]

            try:
                # Compute covariance results for the pair
                results = process_pair(asset1_file, asset2_file, date, start_time)

                # Fill covariance matrices
                cov_matrix_before[i, j] = results["before"]["hayashi_yoshida_covariance"]
                cov_matrix_during[i, j] = results["during"]["hayashi_yoshida_covariance"]
                cov_matrix_after[i, j] = results["after"]["hayashi_yoshida_covariance"]

                # Symmetric matrix: fill the lower triangle
                if i != j:
                    cov_matrix_before[j, i] = cov_matrix_before[i, j]
                    cov_matrix_during[j, i] = cov_matrix_during[i, j]
                    cov_matrix_after[j, i] = cov_matrix_after[i, j]
            except Exception as e:
                print(f"Error processing pair ({asset_names[i]}, {asset_names[j]}): {e}")
                continue

        # Convert covariance matrices to dataframes
        date_label = {"date": [date] * (n_assets * n_assets)}
        col_names = [f"{name}_{name2}" for name in asset_names for name2 in asset_names]

        before_matrix_df = pl.DataFrame({**date_label, **dict(zip(col_names, cov_matrix_before.flatten()))})
        during_matrix_df = pl.DataFrame({**date_label, **dict(zip(col_names, cov_matrix_during.flatten()))})
        after_matrix_df = pl.DataFrame({**date_label, **dict(zip(col_names, cov_matrix_after.flatten()))})

        # Append to the respective parquet files
        before_df = pl.concat([before_df, before_matrix_df], how="vertical").unique(subset=["date"])
        during_df = pl.concat([during_df, during_matrix_df], how="vertical").unique(subset=["date"])
        after_df = pl.concat([after_df, after_matrix_df], how="vertical").unique(subset=["date"])

    # Write the updated parquet files back
    before_df.write_parquet(before_path)
    during_df.write_parquet(during_path)
    after_df.write_parquet(after_path)

    print(f"Covariance matrices saved for all dates in {output_folder}.")

def load_and_reshape_covariance(parquet_path, n_assets=22):

    df = pl.read_parquet(parquet_path)
    reshaped_matrices = {}

    # Convert DataFrame to list of dictionaries
    rows = df.to_dicts()

    for row in rows:
        date = row['date']
        # Extract covariance values by excluding the 'date' key
        cov_flat = [value for key, value in row.items() if key != 'date']
        cov_matrix = np.array(cov_flat).reshape((n_assets, n_assets))
        reshaped_matrices[date] = cov_matrix

    return reshaped_matrices    

def eigenvalue_clipping(matrix):
    """
    Clips the eigenvalues of the input matrix based on a threshold.

    Parameters:
        matrix (ndarray): Input covariance matrix to be processed.

    Returns:
        ndarray: The eigenvalue-clipped matrix.
    """

    # Eigenvalue decomposition
    lambdas, v = eig(matrix)
    N = len(lambdas)

    # Determine q and lambda_plus
    T = matrix.shape[1] if matrix.shape[1] > N else N
    q = N / T
    lambda_plus = (1 + np.sqrt(q))**2

    # Bulk eigenvalues
    sum_lambdas_gt_lambda_plus = np.sum(lambdas[lambdas > lambda_plus])
    sel_bulk = lambdas <= lambda_plus
    N_bulk = np.sum(sel_bulk)
    sum_lambda_bulk = np.sum(lambdas[sel_bulk])
    delta = sum_lambda_bulk / N_bulk  # Average of bulk eigenvalues

    # Modify eigenvalues
    lambdas_clean = lambdas
    lambdas_clean[lambdas_clean <= lambda_plus] = delta

    v_m=np.matrix(v)
    C_clean=np.zeros((N, N))
    for i in range(N-1):    
        C_clean=C_clean+lambdas_clean[i] * np.dot(v_m[i,].T,v_m[i,]) 
        
    np.fill_diagonal(C_clean,1)

    return C_clean

def perform_louvain_clustering(adj_matrix, thresholds=np.linspace(0.12, 0.30, 20)):
    """
    Perform Louvain clustering over a range of thresholds and identify the optimal threshold.

    Args:
        adj_matrix (np.ndarray): The normalized adjacency matrix.
        thresholds (np.ndarray): Array of threshold values to iterate over.

    Returns:
        tuple: (optimal_threshold, min_clusters, partition_optimal)
    """
    # Grid search over thresholds
    threshold = 0.14601202404809618
    results = []

    # Apply threshold
    adj_matrix_thresh = adj_matrix.copy()
    adj_matrix_thresh[adj_matrix_thresh < threshold] = 0

    # Step 3: Create a graph
    G = nx.from_numpy_array(adj_matrix_thresh)

    # Step 4: Perform Louvain clustering
    partition = community.best_partition(G)
    num_clusters = len(set(partition.values()))

    # Store results
    results.append((threshold, num_clusters, partition))

    optimal_result = min(results, key=lambda x: x[1])
    _, _, partition_optimal = optimal_result

    return partition_optimal

def plot_cluster_domains(cluster_domains, clusters_optimal, domain_colors, period, date, save_path):
    """
    Plot pie charts for each cluster showing the distribution of asset domains.

    Args:
        cluster_domains (dict): Mapping of cluster_id to domain counts.
        clusters_optimal (dict): Mapping of cluster_id to list of asset names.
        domain_colors (dict): Mapping of domains to their assigned colors.
        period (str): The period category (before, during, after).
        date (str): The date corresponding to the covariance matrix.
        save_path (str): The file path to save the plot.
    """
    # Get all unique domains
    domains = list(domain_colors.keys())

    # Determine the number of clusters and layout for subplots
    num_clusters = len(clusters_optimal)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_clusters + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  # Flatten in case of multiple rows

    for idx, (cluster_id, domain_count) in enumerate(cluster_domains.items()):
        ax = axes[idx]

        # Prepare data for the pie chart
        labels = []
        sizes = []
        colors = []
        for domain in domains:
            count = domain_count.get(domain, 0)
            if count > 0:
                labels.append(domain)
                sizes.append(count)
                colors.append(domain_colors[domain])

        # Handle clusters with no assets
        if not sizes:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            continue

        # Plot the pie chart with percentage labels
        wedges, texts, autotexts = ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct='%1.1f%%',  # Adds percentage labels
            textprops={'fontsize': 10}
        )
        # Set the title with cluster ID and number of assets
        ax.set_title(
            f'Cluster {cluster_id} ({len(clusters_optimal[cluster_id])} assets)',
            fontsize=14
        )
        ax.axis('equal')  # Ensure pie is a circle

    # Remove any unused subplots
    for i in range(num_clusters, len(axes)):
        fig.delaxes(axes[i])

    # Create a unified legend
    legend_handles = [Patch(facecolor=domain_colors[domain], label=domain) for domain in domains]
    # Position the legend outside the main plot
    fig.legend(handles=legend_handles, title="Domains", loc='upper right', bbox_to_anchor=(0.95, 0.5))

    # Adjust layout and add a main title
    plt.tight_layout(rect=(0, 0, 0.9, 1))  # Convert list to tuple
    plt.suptitle(f'Period: {period.capitalize()}, Date: {date}', fontsize=16, y=1.02)

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

def process_covariance_matrices(cov_matrix_path, output_folder):
    """
    Processes covariance matrices, applies clustering, and generates plots.
    
    Parameters:
        cov_matrix_path (str): Path to the directory containing covariance matrix files.
        output_folder (str): Path to the output directory for saving results.
    """

    # Define the periods and their corresponding file paths
    periods_files = {
        'before': f'{cov_matrix_path}/before.parquet',
        'during': f'{cov_matrix_path}/during.parquet',
        'after': f'{cov_matrix_path}/after.parquet'
    }

    # Create a directory to save all plots
    os.makedirs(output_folder, exist_ok=True)

    # Create separate files for saving clusters for each period
    cluster_output_files = {
        period: os.path.join(output_folder, f'{period}_clusters.jsonl')
        for period in periods_files
    }

    # Ensure the cluster files are empty before appending new results
    for cluster_file in cluster_output_files.values():
        open(cluster_file, 'w').close()

    # Iterate over each period and its corresponding file
    for period, file_path in periods_files.items():
        print(f'\nProcessing Period: {period.capitalize()} | File: {file_path}')

        # Load all covariance matrices for the period
        try:
            cov_matrices_dict = load_and_reshape_covariance(file_path)
        except Exception as e:
            print(f'Error loading file {file_path}: {e}')
            continue

        # Iterate over each date and its covariance matrix
        for date, cov_matrix in cov_matrices_dict.items():
            print(f'  Processing Date: {date}')

            try:
                # Step 1: Apply eigenvalue clipping
                clipped_cov_matrix = eigenvalue_clipping(cov_matrix)
                # Step 2: Fill diagonal with zeros
                np.fill_diagonal(clipped_cov_matrix, 0)
                clipped_cov_matrix = np.asarray(clipped_cov_matrix)
                # Step 3: Normalize the covariance matrix
                scaler = MinMaxScaler()
                adj_matrix = scaler.fit_transform(clipped_cov_matrix)
                # Step 4: Perform Louvain clustering
                partition_optimal = perform_louvain_clustering(adj_matrix)

                clusters_optimal = {}
                for node, cluster_id in partition_optimal.items():
                    # Ensure node index is within the range of asset_names
                    if node < len(asset_names):
                        asset_name = asset_names[node]
                    else:
                        asset_name = f'Asset_{node}'  # Placeholder for undefined assets
                    clusters_optimal.setdefault(cluster_id, []).append(asset_name)

                # Output the optimal clustering results
                print(f'    Optimal Clustering: {clusters_optimal}')

                # Step 5: Save the clusters to the corresponding file
                cluster_data = {
                    'date': date,
                    'clusters': clusters_optimal
                }
                with open(cluster_output_files[period], 'a') as f:
                    f.write(json.dumps(cluster_data) + '\n')

                # Step 6: Process the clusters to count domains
                cluster_domains = {}
                for cluster_id, assets in clusters_optimal.items():
                    # Initialize domain count for the cluster
                    domain_count = {domain: 0 for domain in domain_colors.keys()}
                    for asset in assets:
                        domain = asset_to_domain.get(asset, 'Unknown')
                        domain_count[domain] += 1
                    cluster_domains[cluster_id] = domain_count

                # Step 7: Define the save path for the plot
                # Create a directory for the period if it doesn't exist
                period_dir = os.path.join(output_folder, period)
                os.makedirs(period_dir, exist_ok=True)

                # Define the filename based on the date
                save_filename = f'{period}_{date}.png'
                save_path = os.path.join(period_dir, save_filename)

                # Step 8: Plot and save the cluster domains
                plot_cluster_domains(
                    cluster_domains=cluster_domains,
                    clusters_optimal=clusters_optimal,
                    domain_colors=domain_colors,
                    period=period,
                    date=date,
                    save_path=save_path
                )

                print(f'    Plot saved to: {save_path}')

            except Exception as e:
                print(f'    Error processing date {date}: {e}')
                continue

    print('\nAll clustering and plotting completed.')
