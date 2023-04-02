import pandas as pd
import numpy as np
import os

tickers_path = os.path.join(os.getcwd(), "..", "data", "Tickers")


def tickers_df(data_path, save_path=None) -> pd.DataFrame:
    try:
        (dir, folders, files) = next(os.walk(data_path))
    except StopIteration:
        raise StopIteration("No more suggesstions.")

    tickers_name = [tick[:-4] for tick in files]
    tot_df = None
    for ticker in tickers_name:
        if ticker == "all_data":
            continue
        print(f"Loading ticker: {ticker}")
        df = pd.read_csv(os.path.join(data_path, ticker + ".csv"))
        ticker_name = [ticker for _ in range(len(df))]
        df["ticker"] = ticker_name
        if tot_df is None:
            tot_df = df
        else:
            tot_df = pd.concat([tot_df, df], axis=0, ignore_index=True)

    tot_df = tot_df.set_index([pd.DatetimeIndex(tot_df["datetime"]), "ticker"])
    tot_df.drop("datetime", axis=1, inplace=True)
    tot_df.sort_index(inplace=True)
    return tot_df


def df_to_matrix(df: pd.DataFrame, save_path=None) -> np.array:
    idx = df.index
    levels = idx.nlevels
    maps = [{} for _ in range(levels)]
    idx_np = np.array((*zip(idx.tolist()),)).squeeze()
    unique_idx = [np.sort(np.unique(idx_np[:, i]))[::-1] for i in range(levels)]
    unique_range = [np.arange(len(i)) for i in unique_idx]
    for l in range(levels):
        for i, j in zip(unique_idx[l], unique_range[l]):
            maps[l][i] = j

    max_ids = [i[-1] + 1 for i in unique_range]
    matrix = np.zeros(shape=(*max_ids, len(df.columns)))
    for id, r in zip(idx_np, df.iterrows()):
        np_ids = [maps[i][j] for i, j in enumerate(id)]
        matrix[(*np_ids, None)] = r[1:]

    if save_path:
        np.save(save_path, matrix)
    return matrix, maps


def diff_log(matrix):
    t0 = np.nan_to_num(np.log(matrix[:-1]), nan=0.0, neginf=0.0)
    t1 = np.nan_to_num(np.log(matrix[1:]), nan=0.0, neginf=0.0)
    diff = t1 - t0
    return diff


if __name__ == "__main__":
    tickers_df(tickers_path)
