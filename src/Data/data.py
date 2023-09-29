from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import os
import exchange_calendars as xcals
import tensorflow as tf
from datetime import datetime, timedelta
import pickle
from aiohttp import ClientSession
import asyncio as aio
import string
import yfinance as yf
from bs4 import BeautifulSoup as bso
import logging

Sequence = tf.keras.utils.Sequence

tickers_path = os.path.join(os.getcwd(), "../..", "data", "Tickers")


def yf_to_multiindex(yf_tickers_data: pd.DataFrame) -> dict:
    yf_tickers_data = yf_tickers_data.unstack().reset_index().pivot(index=("Date", "level_0"),
                                                                    columns="level_1").droplevel(0, 1)
    yf_tickers_data.index.names = ["time", "Ticker"]
    yf_tickers_data.columns.name = "columns"
    return yf_tickers_data


async def get_BA_tickers_list(save_path=None) -> dict:
    URL = "https://www.borsaitaliana.it/borsa/azioni/listino-a-z.html?initial="
    alphabet = [string.ascii_uppercase[i] for i in range(26)]
    pages = {}
    tickers = {}

    async def get_tickers_from_page(page, tickers, letter, n):
        if not isinstance(page, bso):
            try:
                page = bso(page, "lxml")
                isins = (page.find("table", {"class": "m-table -firstlevel"})).find_all("a")[2::9]
                for i in isins:
                    try:
                        isin = i["href"].split("/scheda/")[1][:12]
                        name = i["title"].split("Accedi alla scheda strumento\xa0")[1]
                        tickers[isin] = name
                    except Exception as e:
                        print(f"FAILED to process {i} for letter {letter} and page {n}")
            except Exception as e:
                print(f"Letter {letter} and number {n} FAILED to process")
                return

    async def get_page(session: ClientSession, letter: str, pages: dict, number: int):
        resp = await session.get(URL + letter + "&lang=it&page=" + str(number))
        resp_parsed = await resp.read()
        print(f"Response for letter {letter} and page number {number}: {resp.status}")
        try:
            pages[letter][number] = resp_parsed
        except Exception as e:
            pages[letter] = {}
            pages[letter][number] = resp_parsed

    async with ClientSession() as session:
        # Get first page
        tasks = []
        for letter in alphabet:
            tasks.append(get_page(session, letter, pages, 1))
        await aio.gather(*tasks)

        # Get other pages (tot pages)
        tasks = []
        for page_letter in pages.keys():
            phtml = bso(pages[page_letter][1], "lxml")
            try:
                max_n = int(phtml.find_all("ul", {"class": "nav m-pagination__nav"})[0].find_all("a")[-2].text)
                print(f"max page for letter {page_letter} is {max_n}")
                for n in range(2, max_n):
                    tasks.append(get_page(session, page_letter, pages, n))
            except Exception as e:
                print(f"Letter failed to find next page: {page_letter}")
        await aio.gather(*tasks)

    # extract isin and name from pages
    isin_tasks = [get_tickers_from_page(n, tickers, i, j) for i, letter in pages.items() for j, n in letter.items()]
    await aio.gather(*isin_tasks)
    if save_path is not None:
        if not os.path.exists(save_path):
            print(f"Path: {save_path} does not exist, creating...")
            os.makedirs(save_path)
        file_name = os.path.join(save_path, "italian_ISIN.pkl")
        with open(file_name, "wb") as file:
            pickle.dump(tickers, file)
            print(f"Tickers ISIN saved to {file_name}")

    return tickers


def tickers_df(data_path, save_path=None) -> pd.DataFrame:
    try:
        (dir, folders, files) = next(os.walk(data_path))
    except StopIteration as si:
        raise si("No more suggestions, maybe the path does not contain csv files")

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
    """
    Transforms dataframe to multiindex df, must have "datetime":datetime and "tickers":str named columns
    """
    tot_df = df.set_index([pd.DatetimeIndex(df["datetime"]), "ticker"])
    tot_df.drop("datetime", axis=1, inplace=True)
    tot_df.sort_index(inplace=True)
    return tot_df


def df_to_matrix(df: pd.DataFrame,
                 idx_mapping=None,
                 save_path: os.path = None,
                 name: str = "default") -> (np.array, list[dict], list):
    idx = df.index
    levels = idx.nlevels
    idx_np = np.array((*zip(idx.tolist()),)).squeeze()
    unique_idx = [np.sort(np.unique(idx_np[:, i])) for i in range(levels)]

    if idx_mapping is None:
        unique_range = [np.arange(len(i)) for i in unique_idx]
        max_ids = [i[-1] + 1 for i in unique_range]
        maps = [{} for _ in range(levels)]
        for l in range(levels):
            for i, j in zip(unique_idx[l], unique_range[l]):
                maps[l][i] = j
    else:
        assert len(idx_mapping) == levels
        maps = idx_mapping
        max_ids = [np.max(list(i.values())) for i in maps]

    matrix = np.zeros(shape=(*max_ids, len(df.columns)))
    for id, r in zip(idx_np, df.iterrows()):
        np_ids = [maps[i][j] for i, j in enumerate(id)]
        matrix[(*np_ids, None)] = r[1:]

    columns = df.columns

    if save_path:
        try:
            np.save(os.path.join(save_path, name + "_matrix.npy"), matrix)
            np.save(os.path.join(save_path, name + "_unique_index.npy"), unique_idx)
            with open(os.path.join(save_path, name + "_index_mappings.pkl"), "wb") as file:
                pickle.dump(maps, file)
            with open(os.path.join(save_path, name + "_columns_mapping.pkl"), "wb") as file:
                pickle.dump(columns, file)
        except Exception as e:
            raise e
    return matrix, maps, columns


def matrix_to_df(matrix: np.ndarray, idx_mapping: Optional[List[Dict]] = None, columns_mapping=None, **kwargs):
    def generate_names():
        stock_names = []
        counter = 0
        for first in range(52):
            for second in range(52):
                for third in range(52):
                    stock_names.append(first + second + third)
                    counter += 1
                    if counter == matrix.shape[1]:
                        return stock_names

    sparse_matrix = tf.sparse.from_dense(matrix)
    idx = sparse_matrix.indices.numpy()
    column_names = [string.ascii_letters[i] for i in range(matrix.shape[-1])]
    columns = [column_names[i[-1]] for i in idx]
    if idx_mapping is not None:
        idx2time = dict(zip(idx_mapping[0].values(), idx_mapping[0].keys()))
        idx2stock = dict(zip(idx_mapping[1].values(), idx_mapping[1].keys()))
        times = [idx2time[i[0]] for i in idx]
        stocks = [idx2stock[i[1]] for i in idx]
    else:
        start_date = datetime.fromisoformat("2003-09-25")
        end_date = start_date + timedelta(days=matrix.shape[0] - 1)
        time_range = pd.date_range(start_date, end_date, **kwargs)
        times = [time_range[i[0]] for i in idx]
        stock_names = generate_names()
        stocks = [stock_names[i[0]] for i in idx]

    idx_list = [(i, j, k) for i, j, k in zip(times, stocks, columns)]

    multiindex = pd.MultiIndex.from_tuples(idx_list, names=["time", "stock", "columns"])
    df = pd.DataFrame(sparse_matrix.values.numpy(), index=multiindex)
    df = df.unstack(-1).droplevel(0, 1)
    if columns_mapping is not None:
        df.columns = columns_mapping
    return df


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
        self._sequence_len = sequence_len
        self._initial_checks(x, y)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def sequence_len(self):
        return self._sequence_len

    def __len__(self):
        if self.is_input_nested:
            shape = self._x[0].shape[0]
        else:
            shape = self._x.shape[0]
        return shape // self._sequence_len

    def check_len(self):
        if self.is_input_nested:
            shape = self._x[0].shape
        else:
            shape = self._x.shape
        if self._sequence_len > shape[0]:
            raise ValueError("seq_len cannot be greater than the length of the time series")

    def __getitem__(self, idx):
        low = idx * self._sequence_len
        if self.is_input_nested:
            high = tf.math.minimum((idx + 1) * self._sequence_len, self._x[0].shape[0])
        else:
            high = tf.math.minimum((idx + 1) * self._sequence_len, self._x.shape[0])

        lam = lambda i: tf.expand_dims(i[low:high], axis=0)
        if self.is_input_nested:
            next_batch_x = tf.nest.map_structure(lam, self._x)
        else:
            next_batch_x = self._x[low:high][None, :]

        if self.is_output_nested:
            next_batch_y = tf.nest.map_structure(lam, self._y)
        else:
            next_batch_y = self._y[low:high][None, :]
        return next_batch_x, next_batch_y

    def __iter__(self):
        pass

    def on_epoch_end(self):
        pass

    def save(self, path, name="batch_generator"):
        try:
            with open(os.path.join(path, name + "_input.pkl"), "wb") as file:
                pickle.dump(self._x, file)
            with open(os.path.join(path, name + "_output.pkl"), "wb") as file:
                pickle.dump(self._y, file)
        except Exception as e:
            raise e

    def load(self, path):
        try:
            with open(os.path.join(path) + "_input.pkl", "rb") as file:
                x = pickle.load(file)
            with open(os.path.join(path) + "_output.pkl", "rb") as file:
                y = pickle.load(file)
        except Exception as e:
            raise e
        self._initial_checks(x, y)

    def _initial_checks(self, x, y):
        if tf.nest.is_nested(x):
            self._x = [i for i in tf.nest.flatten(x)]
            self.is_input_nested = True
        else:
            self._x = x
            self.is_input_nested = False

        if tf.nest.is_nested(y):
            self._y = tf.nest.flatten(y)
            self.is_output_nested = True
        else:
            self._y = y
            self.is_output_nested = False

        self.check_len()


class StockTimeSeries:
    def __init__(self, data: pd.DataFrame, stock_exchange: str):
        if stock_exchange not in self.available_exchanges:
            raise ValueError(f"Exchange {stock_exchange} not in {self.available_exchanges}")
        self._data = data
        self._stock_exchange = stock_exchange
        self._calendar = xcals.get_calendar(self.stock_exchange)
        self._is_multiindex = None
        self._stocks = None
        self.check_series_index()

    @property
    def available_exchanges(self):
        return xcals.get_calendar_names(include_aliases=True)

    @property
    def stocks(self):
        return self._stocks

    @property
    def is_multiindex(self):
        return self._is_multiindex

    @property
    def data(self):
        return self._data

    @property
    def stock_exchange(self):
        return self._stock_exchange

    @property
    def calendar(self):
        return self._calendar

    def business_days(self, start: datetime, end: datetime):
        data_slice = self.__getitem__(slice(start, end)).data
        if self._is_multiindex:
            index = data_slice.index.unique(0)
        else:
            index = data_slice.index.unique()
        cal = self.calendar.sessions_in_range(index.min(), index.max())
        check = index.isin(cal)
        business_days = index[check]
        mask = np.in1d(data_slice.index.get_level_values(0), business_days)
        return business_days, data_slice, mask

    def remove_non_business_days(self, start=None, end=None, inplace=False):
        business_days, data_slice, mask = self.business_days(start, end)
        new_data = data_slice.loc[mask, :]
        if inplace:
            self._data = new_data
        else:
            return StockTimeSeries(new_data, self.stock_exchange)

    def fill_non_business_days(self, fill_value, start=None, end=None, inplace=True):
        business_days, data_slice, mask = self.business_days(start, end)
        new_data = self._data.loc[mask, :] = fill_value
        if inplace:
            self._data = new_data
        else:
            return StockTimeSeries(new_data, self.stock_exchange)

    def __getitem__(self, item):
        """
        item:
            tuple: tuple(tuple(start rows), tuple(end rows)), tuple(start_column, end_column))
            slice: start_date:end_date
            int: index of datetime row
            string: index of ticker (only for multiindex)
            datetime: row corresponding to date

        Multiindex slice example:
            start_date = datetime.fromisoformat("2005-01-01")
            end_date = datetime.fromisoformat("2006-01-01")
            start_ticker = "A"
            end_ticker = "AAPL"
            start_column = "a"
            end_column = None
            slice_index = ((start_date, start_ticker), (end_date, end_ticker)), (start_column, end_column))

            Internally it gets converted to the following:
            ((slice("2001-01-01", "2002-01-01"), slice("ACN", None), slice(None))
            ((_______datetime slice__________),(__ticker slice__)), (_column slice_)

        """
        datetime_index = self._data.index

        if isinstance(item, slice):
            i, j = item.start, item.stop
            if (isinstance(i, datetime) or i is None) and (isinstance(j, datetime) or j is None):
                if i is None:
                    if self._is_multiindex:
                        i = datetime_index.min()[0]
                    else:
                        i = datetime_index.min()
                if j is None:
                    if self._is_multiindex:
                        j = datetime_index.max()[0]
                    else:
                        j = datetime_index.max()
                if i > j:
                    raise ValueError(f"Start {i} of sequence cannot be greater than End {j}")
                data = self._data.loc[i:j]
            elif (isinstance(i, str) or i is None) and (isinstance(j, str) or j is None):
                assert self._is_multiindex
                return self.__getitem__((((None, i), (None, j)), (None, None)))
            else:
                raise TypeError(f"Slice start {i} and end {j} cannot be of type {type(i)}")
        elif isinstance(item, datetime):
            start = item
            end = item + timedelta(microseconds=1)
            if self._is_multiindex:
                return self.__getitem__((((start, None), (end, None)), (None, None)))
            else:
                return self.__getitem__((((start,), (end,)), (None, None)))
        elif isinstance(item, str):
            assert self._is_multiindex
            return self.__getitem__((((None, item), (None, item + " ")), (None, None)))
        elif isinstance(item, int):
            data = self._data.iloc[item]
        elif isinstance(item, tuple):
            row_slices = item[0]
            row_start_slices = row_slices[0]
            row_end_slices = row_slices[1]
            assert len(row_start_slices) == len(row_end_slices)

            if self._is_multiindex:
                assert len(row_start_slices) > 1 and len(row_end_slices) > 1
                idx_slice = tuple(slice(i, j) for i, j in zip(row_start_slices, row_end_slices))
            else:
                assert len(row_start_slices) == 1 and len(row_end_slices) == 1
                idx_slice = slice(row_start_slices, row_end_slices)

            column_slices = item[1]
            if column_slices is None:
                col_slice = slice(0, None)
            else:
                column_start_slices = column_slices[0]
                column_end_slices = column_slices[1]
                col_slice = slice(column_start_slices, column_end_slices)
            data = self._data.loc[idx_slice, col_slice]
        else:
            raise TypeError(f"Type {type(item)} not supported for indexing")
        return StockTimeSeries(data, self.stock_exchange)

    def __repr__(self):
        return self._data.head(n=np.minimum(len(self._data), 50)).to_string()

    def __len__(self):
        return len(self._data.index)

    def check_series_index(self):
        if isinstance(self._data.index, pd.MultiIndex):
            datetime_index = self._data.index.levels[0]
            assert isinstance(datetime_index, pd.DatetimeIndex)
            self._stocks = list(self._data.index.unique(level=1))
            self._stocks.sort()
            self._is_multiindex = True
        else:
            assert isinstance(self._data.index, pd.DatetimeIndex)
            self._is_multiindex = False
            self._stocks = None

    def to_matrix(self, save_path=None) -> (np.array, list[{}], list):
        if self._is_multiindex:
            return df_to_matrix(self._data, save_path)
        else:
            raise TypeError("Only MultiIndex can be converted to dense matrix")

    def set_data(self, data):
        self._data = data
        self.check_series_index()

    def ptc_change(self, log=False, inplace=False):
        if self.is_multiindex:
            if log:
                data = self._data.groupby(level=1, group_keys=False).apply(lambda x: np.log(1 + x.pct_change()))
            else:
                data = self._data.groupby(level=1, group_keys=False).apply(lambda x: x.pct_change())
        else:
            data = self._data.pct_change()
        data.replace([-np.inf, np.inf, np.nan], 0.0, inplace=True)
        data = data[slice(data.index.get_level_values(0).unique()[1], None)]

        if inplace:
            self._data = data
        else:
            return StockTimeSeries(data, self.stock_exchange)

    def to_time_series_batch_generator(self,
                                       start_train: Optional[datetime] = None,
                                       end_train: Optional[datetime] = None,
                                       start_test: Optional[datetime] = None,
                                       end_test: Optional[datetime] = None,
                                       start_validation: Optional[datetime] = None,
                                       end_validation: Optional[datetime] = None,
                                       adj: Optional[np.ndarray] or pd.DataFrame = None,
                                       seq_len: int = 30,
                                       delta_pred: int = 1,
                                       pred_col: int or str or list[int] or list[str] = -2):
        """
        adj: Adjacency matrix for tickers, None if fully connected (ones matrix), False if not use else ndarray
        """

        def get_x_y(x, delta, idx_mapping=None, pred_col=pred_col):
            cols = self._data.columns
            if isinstance(pred_col, int):
                idx_pred = pred_col
            elif isinstance(pred_col, str):
                try:
                    idx_pred = np.argwhere(cols == pred_col)
                except Exception as e:
                    raise e
            elif isinstance(pred_col, list):
                if all(isinstance(pred_col[i], int) for i in pred_col):
                    idx_pred = pred_col
                elif all(isinstance(pred_col[i], str) for i in pred_col):
                    idx_pred = [np.argwhere(cols == i) for i in pred_col]
                    if np.empty(idx_pred):
                        raise ValueError(f"Columns {pred_col} not found, columns available are {cols}")
                else:
                    raise TypeError(f"List of prediction columns has to be either int or string")
            elif isinstance(pred_col, slice):
                if not isinstance(pred_col.start, int) and isinstance(pred_col.stop, int):
                    idx_pred = pred_col
                if isinstance(pred_col.start, str) and isinstance(pred_col.stop, str):
                    cols = cols.sort_values()
                    start = np.argwhere(cols == pred_col.start)
                    stop = np.argwhere(cols == pred_col.stop)
                    idx_pred = slice(start, stop)
            else:
                raise TypeError(f"Cannot use type {type(pred_col)} for selecting the prediction columns."
                                f"Types can either be int, str, list[int], list[str]")

            if isinstance(x, pd.DataFrame):
                assert delta < x.index.levshape[0] // 2
                x, maps, _ = df_to_matrix(x, idx_mapping=idx_mapping)
                out_ = x[delta:, :, idx_pred]
                if len(out_.shape) == 2:
                    out_ = np.expand_dims(out_, -1)
                in_ = x[:-delta]
                return in_, out_, maps

            elif isinstance(x, np.ndarray):
                assert delta < len(x) // 2 and idx_mapping is None
                out_ = x[delta:, :, idx_pred]
                if len(out_.shape) == 2:
                    out_ = np.expand_dims(out_, -1)
                in_ = x[:-delta]
                return in_, out_

            else:
                raise TypeError(f"Type {type(x)} not supported, only np.array and pd.DataFrame")

        assert self._is_multiindex and self._data.index.get_level_values(
            0).freq != '', "Data is not Multiindex and the datetime index frequency is not defined"
        if start_train is None:
            start_train = self._data.index.min()[0]
        if end_train is None:
            end_train = self._data.index.max()[0]

        assert start_train < end_train

        if start_test is not None:
            if end_test is None:
                assert end_train is not None
            assert end_train < start_test and end_train < end_test and start_validation is None

        if start_validation is not None:
            if end_validation is None:
                assert end_test < start_validation

        tot_data_matrix, maps, cols = self.to_matrix()
        x_train = self.__getitem__(slice(start_train, end_train))
        unique_dates = list(maps[0].keys())
        idx_start_train = np.argwhere(unique_dates == np.asarray(x_train.data.index.min()[0]))[0][0]
        idx_end_train = np.argwhere(unique_dates == np.asarray(x_train.data.index.max()[0]))[0][0]
        x_train, y_train = get_x_y(tot_data_matrix[idx_start_train:idx_end_train], delta_pred)
        train_dates = unique_dates[idx_start_train+delta_pred:idx_end_train]

        if adj:
            if isinstance(adj, bool):
                adj_train = np.ones(shape=(len(x_train), len(self._stocks), len(self._stocks)))
            elif isinstance(adj, np.ndarray):
                assert len(adj.shape) == 3
                times, stocks, _ = adj.shape
                times_, stocks_, _ = self._data.index.levshape
                assert times == times_ and stocks == stocks_
                adj_train = adj[idx_start_train:idx_end_train]
            else:
                raise NotImplementedError()
            train = TimeSeriesBatchGenerator((x_train, adj_train), y_train, seq_len)
        else:
            train = TimeSeriesBatchGenerator(x_train, y_train, seq_len)

        if start_test is not None:
            x_test = self.__getitem__(slice(start_test, end_test))
            idx_start_test = np.argwhere(unique_dates == np.asarray(x_test.data.index.min()[0]))[0][0]
            idx_end_test = np.argwhere(unique_dates == np.asarray(x_test.data.index.max()[0]))[0][0]
            x_test, y_test = get_x_y(tot_data_matrix[idx_start_test:idx_end_test], delta_pred)
            test_dates = unique_dates[idx_start_test+delta_pred:idx_end_test]
            if adj:
                if isinstance(adj, bool):
                    adj_test = np.ones(shape=(len(x_test), len(self._stocks), len(self._stocks)))
                elif isinstance(adj, np.ndarray):
                    adj_test = adj[idx_start_test:idx_end_test]
                else:
                    raise NotImplementedError()
                test = TimeSeriesBatchGenerator((x_test, adj_test), y_test, seq_len)
            else:
                test = TimeSeriesBatchGenerator(x_test, y_test, seq_len)

        else:
            test = None
            test_dates = None

        if start_validation is not None:
            x_validation = self.__getitem__(slice(start_validation, end_validation))
            idx_start_validation = np.argwhere(unique_dates == np.asarray(x_validation.data.index.min())[0])[0][0]
            idx_end_validation = np.argwhere(unique_dates == np.asarray(x_validation.data.index.max())[0])[0][0]
            x_validation, y_validation = get_x_y(tot_data_matrix[idx_start_validation:idx_end_validation], delta_pred)
            validation_dates = unique_dates[idx_start_validation+delta_pred:idx_end_validation]
            if adj:
                if isinstance(adj, bool):
                    adj_validation = np.ones(shape=(len(x_validation), len(self._stocks), len(self._stocks)))
                elif isinstance(adj, np.ndarray):
                    adj_validation = adj[idx_start_validation:idx_end_validation]
                else:
                    raise NotImplementedError()
                validation = TimeSeriesBatchGenerator((x_validation, adj_validation), y_validation, seq_len)
            else:
                validation = TimeSeriesBatchGenerator(x_validation, y_validation, seq_len)
        else:
            validation = None
            validation_dates = None
        return (train, train_dates), (test, test_dates), (validation, validation_dates)

    def save(self, path, name="stock_time_series"):
        try:
            self._data.to_csv(os.path.join(path, name + ".csv"))
        except Exception as e:
            raise e

    def load(self, path):
        try:
            self._data = pd.load_csv(path)
        except Exception as e:
            raise e


if __name__ == "__main__":
    tickers_df(tickers_path)
