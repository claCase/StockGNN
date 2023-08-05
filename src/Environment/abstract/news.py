from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.Environment.abstract.utils import ToDict
from datetime import datetime, timedelta
from src.Environment.abstract.events import Event


class NewsEvent(Event):
    def __init__(self, datetime: datetime, news: str):
        super(NewsEvent, self).__init__("NewsEvent", datetime)
        self._news = news

    @property
    def news(self):
        return self._news
