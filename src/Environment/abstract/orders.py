from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.Environment.abstract.utils import ToDict
from datetime import datetime, timedelta
from src.Environment.abstract.events import Event


@dataclass
class Order:
    minqty: int = 0
    maxprice: float = 0.0
    orderType: str = ''
    action: str = ''
    orderid: int = 0
    exchange: str = ''
    symbol: str = ''
    currency: str = ''


class LimitOrder(Order, ToDict):
    def __init__(self, orderid, exchange, symbol, minqty, maxprice, currency, action, **kwargs):
        super().__init__(orderType="LMT", exchange=exchange, symbol=symbol, orderid=orderid, minqty=minqty,
                         maxprice=maxprice, currency=currency, action=action, **kwargs)


class MarketOrder(Order, ToDict):
    def __init__(self, orderid, exchange, symbol, minqty, action, currency, **kwargs):
        super().__init__(orderType="MKT", orderid=orderid, exchange=exchange, symbol=symbol, minqty=minqty,
                         action=action, currency=currency, **kwargs)


class OrderEvent(Event):
    def __init__(self, order: Order, datetime: datetime):
        super().__init__("ORDER", datetime)
        self._order = order

    @property
    def order(self):
        return self._order


class FillEvent(Event):
    def __init__(self, order: Order, quantity: int, price: float, datetime: datetime):
        super().__init__("FILL", datetime=datetime)
        self._order = order
        self._filledQty = quantity
        self._filledPrice = price

    @property
    def order(self) -> Order:
        return self._order

    @property
    def fill_quantity(self) -> int:
        return self._filledQty

    @property
    def fill_price(self) -> float:
        return self._filledPrice

    @property
    def transaction_cost(self) -> float:
        raise NotImplementedError("Implement transaction cost calculation based on the order")


class OrderState:
    def __init__(self, order):
        self._order: Order = order
        self._status: str = ''
        self._beginReqTime: datetime = datetime.now()
        self._endReqTime: datetime = self._beginReqTime + timedelta(10000)
        self._fills: [FillEvent] = []

    def update(self, fillEvent: FillEvent):
        self._fills.append(fillEvent)
        if fillEvent.type == "FUFILLED":
            self._endReqTime = datetime.now()

    @property
    def order(self) -> Order:
        return self._order

    @property
    def status(self) -> str:
        return self._status

    @property
    def begin_time(self) -> datetime:
        return self._beginReqTime

    @property
    def end_time(self) -> datetime:
        return self._endReqTime

    @property
    def fill_events(self) -> [FillEvent]:
        return self._fills

    def quantity_to_fill(self) -> int:
        quantity = self.order.minqty
        for event in self._fills:
            quantity -= event.fill_quantity
        return quantity
