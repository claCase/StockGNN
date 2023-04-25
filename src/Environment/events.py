import pandas as pd
import abc
from datetime import datetime
from typing import Optional


class Event(abc.ABC):
    def __init__(self, type: str, datetime: datetime):
        self._type = type
        self._datetime = datetime

    @property
    def datetime(self):
        return self._datetime

    @property
    def type(self):
        return self._type


class StockPriceQuantityEvent(Event):
    def __init__(self, symbol: str, datetime: datetime, quantity: float, price: Optional[float] = None):
        super(StockPriceQuantityEvent, self).__init__("UpdatePriceQuantity", datetime)
        self._symbol = symbol
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


class NewsEvent(Event):
    def __init__(self, datetime: datetime, news: str):
        super(NewsEvent, self).__init__("NewsEvent", datetime)
        self._news = news

    @property
    def news(self):
        return self._news


class OrderEvent(StockPriceQuantityEvent):
    def __init__(self, symbol: str, exchange: str, datetime: datetime, price: float, quantity: float):
        super().__init__(symbol, datetime, quantity, price)
        self._exchange = exchange

    @property
    def exchange(self):
        return self._exchange


class LimitOrderEvent(OrderEvent):
    def __init__(self, symbol: str, exchange: str, datetime: datetime, price: float, quantity: float):
        super().__init__(symbol, exchange, datetime, quantity, price)


class MarketOrderEvent(OrderEvent):
    def __init__(self, symbol: str, exchange: str, datetime: datetime, quantity: float):
        super().__init__(symbol, exchange, datetime, quantity)


class FillEvent(OrderEvent):
    def __init__(self, symbol: str, exchange: str, order: OrderEvent, datetime: datetime, quantity: float,
                 price: float):
        super().__init__(symbol, exchange, datetime, quantity, price)
        self._oder = order

    @property
    def order(self):
        return self._oder

    def calculate_filling_costs(self):
        raise NotImplementedError
