from enum import Enum
import numpy as np

from src.Environment.abstract.events import (Event,
                                             DataProcessEvent,
                                             DataStoreEvent,
                                             GatherEvent,
                                             DATA_GATHER_MESSAGES,
                                             DATA_PROCESS_MESSAGES,
                                             DATA_STORE_MESSAGES)
from datetime import datetime, timedelta
import pandas as pd


class CSVEventGather(GatherEvent):
    def __init__(self, df: pd.DataFrame, msg, datetime):
        super(CSVEventGather, self).__init__(msg=msg, datetime=datetime, data=df)

    def check_message_code(self):
        assert self.message in DATA_GATHER_MESSAGES


class NPYEventGather(GatherEvent):
    def __init__(self, arr: np.array, msg, datetime):
        super(NPYEventGather, self).__init__(msg=msg, datetime=datetime, data=arr)

    def check_message_code(self):
        assert self.message in DATA_GATHER_MESSAGES


class CSVEventSave(DataStoreEvent):
    def __init__(self, df: pd.DataFrame, msg, datetime):
        super(CSVEventSave, self).__init__(msg=msg, datetime=datetime, data=df)

    def check_message_code(self):
        assert self.message in DATA_STORE_MESSAGES


class CSVEventProcess(DataProcessEvent):
    def __init__(self, data, msg, datetime):
        super(CSVEventProcess, self).__init__(msg=msg, datetime=datetime, data=data)

    def check_message_code(self):
        assert self.message in DATA_PROCESS_MESSAGES


class NumpyEventSave(DataStoreEvent):
    def __init__(self, df: pd.DataFrame, msg, datetime):
        super(NumpyEventSave, self).__init__(msg=msg, datetime=datetime, data=df)

    def check_message_code(self):
        assert self.message in DATA_STORE_MESSAGES


class StockTimeseriesDataFrame(DataProcessEvent):
    def __init__(self, df: pd.DataFrame, msg, datetime):
        super().__init__(msg, df, datetime)


class StockTimeSeriesSave(DataStoreEvent):
    def __init__(self, df: pd.DataFrame, msg, datetime):
        super(StockTimeSeriesSave, self).__init__(msg=msg, datetime=datetime, data=df)

    def check_message_code(self):
        assert self.message in DATA_STORE_MESSAGES


class StockTimeSeriesProcess(DataProcessEvent):
    def __init__(self, df: pd.DataFrame, msg, datetime):
        super(StockTimeSeriesProcess, self).__init__(msg=msg, datetime=datetime, data=df)

    def check_message_code(self):
        assert self.message in DATA_STORE_MESSAGES


class StockTimeSeriesConvert(DataProcessEvent):
    def __init__(self, df: pd.DataFrame, msg, datetime):
        super(StockTimeSeriesConvert, self).__init__(msg=msg, datetime=datetime, data=df)

    def check_message_code(self):
        assert self.message in DATA_STORE_MESSAGES


class StockPriceQuantityEvent(Event):
    def __init__(self, symbol: str, exchange: str, datetime: datetime, quantity: float, price: float):
        super(StockPriceQuantityEvent, self).__init__("UPDPQ", datetime)
        self._symbol = symbol
        self._exchange = exchange
        self._price = price
        self._quantity = quantity

    @property
    def price(self):
        return self._price

    @property
    def quantity(self):
        return self._quantity

    @property
    def symbol(self):
        return self._symbol

    @property
    def exchange(self):
        return self._exchange
