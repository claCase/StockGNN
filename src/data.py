import pandas as pd
import numpy as np
import os
import exchange_calendars as excal
import datetime
import tensorflow as tf

Sequence = tf.keras.utils.Sequence

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
    if save_path is not None:
        print(f"Saving Dataframe to: {save_path}")
        tot_df.to_csv(save_path)
    return tot_df


def df_to_matrix(df: pd.DataFrame, save_path=None) -> (np.array, ({})):
    idx = df.index
    levels = idx.nlevels
    maps = [{} for _ in range(levels)]
    idx_np = np.array((*zip(idx.tolist()),)).squeeze()
    unique_idx = [np.sort(np.unique(idx_np[:, i])) for i in range(levels)]
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


class TimeSeries:
    def __init__(self, series: pd.DataFrame):
        self.series = series
        self.index = series.index

    def slice(self, start, end):
        raise NotImplementedError()


class FinancialTimeSeries(TimeSeries):
    def __len__(self, stock_exchange, **kwargs):
        super().__init__(**kwargs)
        self.stock_exchange = stock_exchange

    def check_business_days(self,
                            start: datetime.datetime,
                            end: datetime.datetime = None):
        pass


class TimeSeriesBatchGenerator(Sequence):
    '''
    Used in conjunction with stateful rnn to keep the previous batch ending hidden state. It returns batches of time slices
    of the time series. The batches are not randomly shuffled to preserve the time order of the time series
    args:
    x: features time series TxNxf or nested tensor (tuple) of feature time series and adj matrix of shape (TxNxf, TxNxN)
    y: time series targets of shape TxNxf2
    sequence_len
    '''

    def __init__(self, x, y, sequence_len):
        super().__init__()
        if tf.nest.is_nested(x):
            self.x = [i for i in tf.nest.flatten(x)]
            self.is_input_nested = True
        else:
            self.x = x
            self.is_input_nested = False

        if tf.nest.is_nested(y):
            self.y = tf.nest.flatten(y)
            self.is_output_nested = True
        else:
            self.y = y
            self.is_output_nested = False

        self.sequence_len = sequence_len
        self.check_len()

    def __len__(self):
        if self.is_input_nested:
            shape = self.x[0].shape[0]
        else:
            shape = self.x.shape[0]
        return shape // self.sequence_len

    def check_len(self):
        if self.is_input_nested:
            shape = self.x[0].shape
        else:
            shape = self.x.shape
        if self.sequence_len > shape[0]:
            raise ValueError("seq_len cannot be greater than the length of the time series")

    def __getitem__(self, idx):
        low = idx * self.sequence_len
        if self.is_input_nested:
            high = tf.math.minimum((idx + 1) * self.sequence_len, self.x[0].shape[0])
        else:
            high = tf.math.minimum((idx + 1) * self.sequence_len, self.x.shape[0])

        if self.is_input_nested:
            lam = lambda i: tf.expand_dims(i[low:high], axis=0)
            next_batch_x = tf.nest.map_structure(lam, self.x)
            next_batch_y = tf.nest.map_structure(lam, self.y)
        else:
            next_batch_x = self.x[low:high][None, :]
            next_batch_y = self.y[low:high][None, :]
        return next_batch_x, next_batch_y

    def __iter__(self):
        pass

    def on_epoch_end(self):
        pass


if __name__ == "__main__":
    tickers_df(tickers_path)
