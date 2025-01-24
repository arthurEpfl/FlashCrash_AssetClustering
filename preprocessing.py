import tarfile
import polars as pl
import time
import os

''' 
All funtions below are a generalization to all stocks of the functions defined in notebook preprocessing.ipynb. 
To run the file preprocessing.py, it takes about 1 hour on local machine.
We start from the files produced here for the rest of the project.
'''



def load_bbo(filename,
            tz_exchange="America/New_York",
            only_regular_trading_hours=True,
            hhmmss_open="09:30:00",
            hhmmss_close="16:00:00",
            merge_same_index=True):
    try:
        if filename.endswith("csv") or filename.endswith("csv.gz"):
            DF=pl.read_csv(filename)
        elif filename.endswith("parquet"):    
            DF=pl.read_parquet(filename)
        else:
            print("cannot load file "+filename+" : unknown format")
            return None
        if DF.is_empty():
            print(f"Skipping file: {filename} (empty file)")
            return None
    except:
        print(filename+" cannot be loaded")
        return None

    excel_base_date = pl.datetime(1899, 12, 30)  # Excel starts counting from 1900-01-01, but Polars needs 1899-12-30
    DF = DF.with_columns(
        (pl.col("xltime") * pl.duration(days=1) + excel_base_date).alias("index")
    )
    DF = DF.with_columns(pl.col("index").dt.convert_time_zone(tz_exchange))
    DF.drop("xltime")

    # apply common sense filter
    DF = DF.filter(pl.col("ask-price")>0).filter(pl.col("bid-price")>0).filter(pl.col("ask-price")>pl.col("bid-price"))

    if merge_same_index:
        DF = DF.group_by('index',maintain_order=True).last()   # last quote of the same timestamp
    
    if only_regular_trading_hours:
        hh_open,mm_open,ss_open = [float(x) for x in hhmmss_open.split(":")]
        hh_close,mm_close,ss_close = [float(x) for x in hhmmss_close.split(":")]

        seconds_open=hh_open*3600+mm_open*60+ss_open
        seconds_close=hh_close*3600+mm_close*60+ss_close

        DF = DF.filter(pl.col('index').dt.hour().cast(float)*3600+pl.col('index').dt.minute().cast(float)*60+pl.col('index').dt.second()>=seconds_open,
                       pl.col('index').dt.hour().cast(float)*3600+pl.col('index').dt.minute().cast(float)*60+pl.col('index').dt.second()<=seconds_close)
    
    return DF

def load_trade(filename,
            tz_exchange="America/New_York",
            only_non_special_trades=True,
            only_regular_trading_hours=True,
            merge_sub_trades=True):
    try:
        if filename.endswith("csv") or filename.endswith("csv.gz"):
            DF=pl.read_csv(filename, ignore_errors=True)
        elif filename.endswith("parquet"):    
            DF=pl.read_parquet(filename)
        else:
            print("cannot load file "+filename+" : unknown format")
            return None
        if DF.is_empty():
            print(f"Skipping file: {filename} (empty file)")
            return None
    except:
        print(filename+" cannot be loaded")
        return None

    excel_base_date = pl.datetime(1899, 12, 30)  # Excel starts counting from 1900-01-01, but Polars needs 1899-12-30
    DF = DF.with_columns(
        (pl.col("xltime") * pl.duration(days=1) + excel_base_date).alias("index")
    )
    DF = DF.with_columns(pl.col("index").dt.convert_time_zone(tz_exchange))
    DF.drop(["xltime","trade-rawflag","trade-stringflag"])

    if only_non_special_trades:
        DF=DF.filter(pl.col("trade-stringflag")=="uncategorized")

    if DF["trade-price"].dtype != pl.Float64:
                    DF = DF.with_columns(pl.col("trade-price").cast(pl.Float64))
    if DF["trade-volume"].dtype != pl.Float64:
                    DF = DF.with_columns(pl.col("trade-volume").cast(pl.Float64))

    if merge_sub_trades:   # average volume-weighted trade price here
        DF=DF.group_by('index',maintain_order=True).agg([(pl.col('trade-price')*pl.col('trade-volume')).sum()/(pl.col('trade-volume').sum()).alias('trade-price'),pl.sum('trade-volume')])        
    
    return DF


tar_paths = ['AAPL.OQ-2010.tar', 'AMGN.OQ-2010.tar', 'AXP.N-2010.tar', 'BA.N-2010.tar', 'CAT.N-2010.tar', 'CSCO.OQ-2010.tar', 'CVX.N-2010.tar', 'DOW.N-2010.tar', 'GS.N-2010.tar', 'HD.N-2010.tar', 'IBM.N-2010.tar', 'INTC.OQ-2010.tar', 'JNJ.N-2010.tar', 'JPM.N-2010.tar', 'KO.N-2010.tar', 'MCD.N-2010.tar', 'MMM.N-2010.tar', 'MRK.N-2010.tar', 'MSFT.OQ-2010.tar', 'NKE.N-2010.tar', 'PFE.N-2010.tar', 'PG.N-2010.tar', 'RTX.N-2010.tar', 'TRV.N-2010.tar', 'UNH.N-2010.tar', 'UTX.N-2010.tar', 'V.N-2010.tar', 'VZ.N-2010.tar', 'WBA.OQ-2010.tar', 'WMT.N-2010.tar', 'XOM.N-2010.tar']
extract_path = './extracted_files'

def get_stocks_names(paths_list):
    stocks = []
    for path in paths_list:
        filename = os.path.basename(path)
        stock = filename.split('-')[0]
        stocks.append(stock)
    return stocks


def extract_files(tar_paths, extract_path):
    for tar_path in tar_paths:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_path)

def load_bbo_trade_files(stocks):
    base_input_dir = "extracted_files/data/extraction/TRTH/raw/equities/US"
    output_base_dir = "processed"

    stocks = stocks

    categories = {
        "bbo": "bbo",
        "trade": "trade"
    }
    
    print(f'categories:{categories}')
    for stock in stocks:
        for category, subdirectory in categories.items():
            os.makedirs(os.path.join(output_base_dir, category, stock), exist_ok=True)
            input_dir = os.path.join(base_input_dir, subdirectory, f"{stock}")
            output_dir = os.path.join(output_base_dir, category, stock)

            file_counter = 0 

            for root, _, files in os.walk(input_dir):
                for file in sorted(files):
                    if file.endswith(".csv.gz"):
                        file_counter += 1
                        if file_counter == 1:
                            continue

                        input_path = os.path.join(root, file)

                        if category == "bbo":
                            processed_data = load_bbo(input_path)
                        elif category == "trade":
                            processed_data = load_trade(input_path)
                        else:
                            continue

                        if processed_data is not None:
                            output_file = file.replace(".csv.gz", ".csv")
                            output_path = os.path.join(output_dir, output_file)
                            processed_data.write_csv(output_path)

def merge_trade_bbo(stocks):
    """
    Merge trade and bbo files for multiple stocks, joining on matching dates.
    """
    base_bbo_dir = "processed/bbo"
    base_trade_dir = "processed/trade"
    base_joined_dir = "processed/joined"

    def extract_date(filename, suffix):
        if filename.endswith(suffix):
            parts = filename.split("-")
            if len(parts) >= 3:
                return "-".join(parts[:3])
        return None

    for stock in stocks:
        print(f"\nProcessing stock: {stock}")

        bbo_dir = os.path.join(base_bbo_dir, stock)
        trade_dir = os.path.join(base_trade_dir, stock)
        joined_dir = os.path.join(base_joined_dir, stock)

        os.makedirs(joined_dir, exist_ok=True)

        bbo_dates = []
        for file in os.listdir(bbo_dir):
            if file.endswith("-bbo.csv"):
                date = extract_date(file, "-bbo.csv")
                if date:
                    bbo_dates.append((date, file))

        trade_dates = []
        for file in os.listdir(trade_dir):
            if file.endswith("-trade.csv"):
                date = extract_date(file, "-trade.csv")
                if date:
                    trade_dates.append((date, file))

        bbo_dict = {date: file for date, file in bbo_dates}
        trade_dict = {date: file for date, file in trade_dates}

        common_dates = set(bbo_dict.keys()).intersection(trade_dict.keys())

        for date in common_dates:
            bbo_file = os.path.join(bbo_dir, bbo_dict[date])
            trade_file = os.path.join(trade_dir, trade_dict[date])

            try:
                bbo_df = pl.read_csv(bbo_file)
                trade_df = pl.read_csv(trade_file)

                bbo_df = bbo_df.with_columns(
                    pl.col("index")
                    .str.replace(r"[-+]\d{4}$", "")
                    .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.6f", strict=False)
                    .alias("index")
                )

                trade_df = trade_df.with_columns(
                    pl.col("index")
                    .str.replace(r"[-+]\d{4}$", "")
                    .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.6f", strict=False)
                    .alias("index")
                )

                events = bbo_df.join(trade_df, on='index', how="full", coalesce=True).sort('index')
                events = events.with_columns(pl.col('index').dt.date().alias('date'))
                events = events.with_columns([
                    pl.col("bid-price").forward_fill().over([pl.col("date")]),
                    pl.col("bid-volume").forward_fill().over([pl.col("date")]),
                    pl.col("ask-price").forward_fill().over([pl.col("date")]),
                    pl.col("ask-volume").forward_fill().over([pl.col("date")]),
                    pl.col("trade-price").forward_fill().over([pl.col("date")]),
                    pl.col("trade-volume").forward_fill().over([pl.col("date")])
                ])

                events_file = os.path.join(joined_dir, f"{date}-{stock}.OQ-joined.csv")
                events.write_csv(events_file)
                print(f"Joined and saved: {events_file}")
            except Exception as e:
                print(f"Error processing files for date {date} ({stock}): {e}")

        print(f"Total number of joined files for {stock}: {len(common_dates)}")

def convert_to_parquet(stocks):
    """
    Ensure all files have the same data types and save them as Parquet, then remove original CSVs.
    """
    base_joined_dir = "processed/joined"

    for stock in stocks:
        print(f"\nProcessing stock: {stock}")

        joined_dir = os.path.join(base_joined_dir, stock)


        files = [os.path.join(joined_dir, file) for file in os.listdir(joined_dir) if file.endswith(".csv")]

        for file in sorted(files):
            try:
                df = pl.read_csv(file)

                df = df.with_columns(pl.col("trade-price").fill_null("0.0"))
                df = df.with_columns(pl.col("trade-volume").fill_null("0"))

                if df["trade-price"].dtype != pl.Float64:
                    df = df.with_columns(pl.col("trade-price").cast(pl.Float64))

                if df["trade-volume"].dtype != pl.Int64:
                    df = df.with_columns(pl.col("trade-volume").cast(pl.Float64))
                    df = df.with_columns(pl.col("trade-volume").cast(pl.Int64))

                df = df.with_columns(
                    pl.when(pl.col("trade-price") == 0.0).then(None).otherwise(pl.col("trade-price")).alias("trade-price")
                )
                df = df.with_columns(
                    pl.when(pl.col("trade-volume") == 0).then(None).otherwise(pl.col("trade-volume")).alias("trade-volume")
                )

                parquet_file = file.replace(".csv", ".parquet")
                df.write_parquet(parquet_file)

                os.remove(file)

            except Exception as e:
                print(f"Error processing file {file} for stock {stock}: {e}")

        print(f"Processing complete for stock: {stock}")

def merge_yearly_data(stocks):
    """
    Merge all files for a given year for multiple stocks into a single DataFrame and save as Parquet.
    """
    base_joined_dir = "processed/joined"
    base_output_dir = "processed/final_yearly"

    os.makedirs(base_output_dir, exist_ok=True)

    for stock in stocks:
        print(f"\nProcessing stock: {stock}")

        joined_dir = os.path.join(base_joined_dir, stock)
        output_file = os.path.join(base_output_dir, f"{stock}_combined.parquet")

        files = [os.path.join(joined_dir, file) for file in os.listdir(joined_dir) if file.endswith(".parquet")]

        combined_df = pl.DataFrame()

        for file in sorted(files):
            try:
                df = pl.read_parquet(file)

                combined_df = pl.concat([combined_df, df], how="vertical")
            except Exception as e:
                print(f"Error reading file {file} for stock {stock}: {e}")

        if combined_df.is_empty():
            print(f"No data to save for stock {stock}. Skipping.")
            continue

        combined_df = combined_df.sort("index")

        try:
            combined_df.write_parquet(output_file)
            print(f"Combined data saved for stock {stock}: {output_file}")
        except Exception as e:
            print(f"Error saving combined file for stock {stock}: {e}")

def run_all():
    '''
    A run of all the functions to get the final preprocessed data.
    '''
    tar_paths = ['AAPL.OQ-2010.tar', 'AMGN.OQ-2010.tar', 'AXP.N-2010.tar', 'BA.N-2010.tar', 'CAT.N-2010.tar', 'CSCO.OQ-2010.tar', 'CVX.N-2010.tar', 'DOW.N-2010.tar', 'GS.N-2010.tar', 'HD.N-2010.tar', 'IBM.N-2010.tar', 'INTC.OQ-2010.tar', 'JNJ.N-2010.tar', 'JPM.N-2010.tar', 'KO.N-2010.tar', 'MCD.N-2010.tar', 'MMM.N-2010.tar', 'MRK.N-2010.tar', 'MSFT.OQ-2010.tar', 'NKE.N-2010.tar', 'PFE.N-2010.tar', 'PG.N-2010.tar', 'RTX.N-2010.tar', 'TRV.N-2010.tar', 'UNH.N-2010.tar', 'UTX.N-2010.tar', 'V.N-2010.tar', 'VZ.N-2010.tar', 'WBA.OQ-2010.tar', 'WMT.N-2010.tar', 'XOM.N-2010.tar']
    extract_path = 'extracted_files'
    print('Running')

    stock_names = get_stocks_names(tar_paths)

    extract_files(tar_paths, extract_path)
    load_bbo_trade_files(stock_names)
    merge_trade_bbo(stock_names)
    convert_to_parquet(stock_names)
    merge_yearly_data(stock_names)

run_all()