from dataclasses import dataclass
import pandas as pd
import abc
from datetime import datetime
from typing import Optional, Literal
import json
from src.Environment.abstract.utils import ToDict


class Event(abc.ABC, ToDict):
    def __init__(self, type: str, datetime: datetime, **kwargs):
        super(ToDict, self).__init__(**kwargs)
        self._type = type
        self._datetime = datetime

    @property
    def datetime(self):
        return self._datetime

    @property
    def type(self):
        return self._type


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
