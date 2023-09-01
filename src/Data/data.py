from typing import Optional, List
import pandas as pd
import numpy as np
import os
import exchange_calendars as xcals
import tensorflow as tf
import datetime
import string

Sequence = tf.keras.utils.Sequence

tickers_path = os.path.join(os.getcwd(), "../..", "data", "Tickers")


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
    tot_df = to_multiindex(tot_df)
    if save_path is not None:
        print(f"Saving Dataframe to: {save_path}")
        tot_df.to_csv(save_path)
    return tot_df


def to_multiindex(df: pd.DataFrame):
    tot_df = df.set_index([pd.DatetimeIndex(df["datetime"]), "ticker"])
    tot_df.drop("datetime", axis=1, inplace=True)
    tot_df.sort_index(inplace=True)
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
        np.save(os.path.join(save_path, "matrix.npy"), matrix)
        np.save(os.path.join(save_path, "unique_index_mapping.npy"), unique_idx)
    return matrix, maps


def matrix_to_df(matrix: np.ndarray, index_mapping: Optional[List[np.ndarray]] = None, **kwargs):
    if index_mapping is None:
        start_date = datetime.date.fromisoformat("2000-01-01")
        end_date = start_date + datetime.timedelta(days=matrix.shape[0] - 1)
        stock_names = []
        counter = 0
        for first in range(52):
            for second in range(52):
                for third in range(52):
                    stock_names.append(first + second + third)
                    counter += 1
                    if counter == matrix.shape[1]:
                        break
                else:
                    continue
                break
            else:
                continue
            break

    else:
        start_date = index_mapping[0].min()
        end_date = index_mapping[0].max()
        stock_names = index_mapping[1]

    time_index = pd.date_range(start_date, end_date, **kwargs)
    assert time_index.shape == matrix.shape[0]
    multiindex = pd.MultiIndex.from_product([time_index, stock_names], names=["time", "stock"])
    df = pd.DataFrame(matrix.reshape(-1, 5), index=multiindex, columns=[string.ascii_letters[i] for i in range(5)])
    return df


class StockTimeSeries:
    def __init__(self, data: pd.DataFrame, stock_exchange: str):
        if stock_exchange not in self.available_exchanges:
            raise ValueError(f"Exchange {stock_exchange} not in {self.available_exchanges}")
        self.data = data
        self.stock_exchange = stock_exchange
        self.calendar = xcals.get_calendar(self.stock_exchange)
        self.check_series_index()

    def is_business_day(self, start: datetime.datetime, end: datetime.datetime):
        data_slice = self.__getitem__(slice(start, end)).data
        if self._is_multiindex:
            index = pd.DatetimeIndex(np.asarray(data_slice.index.to_list())[:, 0])
        else:
            index = data_slice.index
        cal = self.calendar.sessions_in_range(index.min(), index.max())
        check = index.isin(cal)
        return check

    def remove_non_business_days(self, start=None, end=None, inplace=True):
        new_data = self.__getitem__(slice(start, end)).data.iloc[self.is_business_day(start, end)]
        if inplace:
            self.data = new_data
        else:
            return StockTimeSeries(new_data, self.stock_exchange)

    def fill_non_business_days(self, fill_value, start=None, end=None, inplace=True):
        is_business_day = self.is_business_day(start, end)
        new_data = self.data[is_business_day] = fill_value
        if inplace:
            self.data = new_data
        else:
            return StockTimeSeries(new_data, self.stock_exchange)

    def __getitem__(self, item):
        if self._is_multiindex:
            datetime_index = np.asarray(self.data.index.to_list())[:, 0]
        else:
            datetime_index = self.data.index
        if isinstance(item, slice):
            i, j = item.start, item.stop
            if i is None:
                i = datetime_index.min()
            if j is None:
                j = datetime_index.max()
            if i > j:
                raise ValueError(f"Start {i} of sequence cannot be greater than End {j}")
            data = self.data.iloc[(i <= datetime_index) & (datetime_index <= j)]
        elif isinstance(item, pd.DatetimeIndex):
            data = self.data.iloc[datetime_index == item]
        elif isinstance(item, int):
            data = self.data.iloc[item]
        else:
            raise TypeError(f"Type {type(item)} not supported for indexing")
        return StockTimeSeries(data, self.stock_exchange)

    def __repr__(self):
        return self.data.head(n=np.minimum(len(self.data), 50)).to_string()

    @property
    def available_exchanges(self):
        return xcals.get_calendar_names(include_aliases=False)

    def __len__(self):
        return len(self.data.index)

    def check_series_index(self):
        if isinstance(self.data.index, pd.MultiIndex):
            datetime_index = self.data.index.levels[0]
            assert isinstance(datetime_index, pd.DatetimeIndex)
            self.stocks = list(self.data.index.levels[1]).sort()
            self._is_multiindex = True
        else:
            assert isinstance(self.data.index, pd.DatetimeIndex)
            self._is_multiindex = False
            self.stocks = None

    def to_matrix(self, save_path) -> (np.array, {}):
        if self._is_multiindex:
            return df_to_matrix(self.data, save_path)
        else:
            raise TypeError("Only MultiIndex can be converted to dense matrix")


class TimeSeriesBatchGenerator(Sequence):
    '''
    Used in conjunction with stateful rnn to keep the previous batch ending hidden state. It returns batches of time
    slices of the time series. The batches are not randomly shuffled to preserve the time order of the time series
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
