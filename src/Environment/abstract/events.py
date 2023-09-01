from dataclasses import dataclass
import pandas as pd
from abc import ABC
from datetime import datetime
from abc import ABC, abstractmethod
# from typing import Optional, Literal
import json
from src.Environment.abstract.utils import ToDict
from enum import Enum


class DATA_STORE_MESSAGES(Enum):
    SAVING = "SAVING"
    SAVED = "SAVED"
    SAVE_FAILED = "SAVE_FAILED"


class DATA_PROCESS_MESSAGES(Enum):
    PROCESSED = "PROCESSED"
    PROCESSING = "PROCESSING"
    PROCESS_FAILED = "PROCESS_FAILED"


class DATA_GATHER_MESSAGES(Enum):
    CONNECTING = "CONNECTING"
    DISCONNECTING = "DISCONNECTING"
    GATHERING = "GATHERING"
    GATHERED = "GATHERED"
    GATHER_FAILED = "GATHER_FAILED"


class Event(ABC, ToDict):
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


class MaximumTimeExceeded(Event):
    def __init__(self, name, datetime):
        super(MaximumTimeExceeded, self).__init__(type="MAXTIME", datetime=datetime)
        self._name = name

    @property
    def name(self):
        return self._name


class DataEvent(Event):
    def __init__(self, type, msg, data, datetime: datetime):
        super().__init__(type=type, datetime=datetime)
        self._msg = msg
        self._data = data

    @property
    def message(self):
        return self._msg

    @property
    def data(self):
        return self._data

    @abstractmethod
    def check_message_code(self):
        raise NotImplementedError("Implement checking logic")


class GatherEvent(DataEvent):
    def __init__(self, msg, data, datetime: datetime):
        super().__init__(type="DG", msg=msg, datetime=datetime, data=data)

    def check_message_code(self):
        assert self.message in DATA_GATHER_MESSAGES


class DataStoreEvent(DataEvent):
    def __init__(self, msg, data, datetime: datetime):
        super().__init__(type="DS", msg=msg, data=data, datetime=datetime)

    def check_message_code(self):
        assert self.message in DATA_STORE_MESSAGES


class DataProcessEvent(DataEvent):
    def __init__(self, msg, data, datetime: datetime):
        super().__init__(type="DP", msg=msg, data=data, datetime=datetime)

    def check_message_code(self):
        assert self.message in DATA_PROCESS_MESSAGES
