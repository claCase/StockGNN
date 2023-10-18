import numpy as np
import pandas as pd
from datetime import datetime
from ..abstract.data_handlers import Producer
from ..abstract.events import GatherEvent, DATA_GATHER_MESSAGES
from ..modules.events import CSVEventGather, NPYEventGather
from ...Modelling.data import tickers_df


class CSVLoader(Producer):
    def __init__(self, name, data_path, **kwargs):
        super().__init__(type="CSV", name=name, **kwargs)
        self._data_path = data_path
        self._data = None

    @property
    def data_path(self):
        return self.data_path

    async def load_data(self):
        if self._data_path[-3:] == "csv":
            df = pd.read_csv(self._data_path)
        else:
            try:
                df = tickers_df(self._data_path)
            except Exception as e:
                raise e
        return df

    async def _gather_data(self):
        try:
            self._data = await self.load_data()
            self._is_done = True
            return CSVEventGather(self._data, datetime=datetime.now(), msg=DATA_GATHER_MESSAGES("GATHERED"))
        except IOError as ioe:
            return GatherEvent(msg=DATA_GATHER_MESSAGES("FAILED"), datetime=datetime.now(), data=None)
        except Exception as e:
            raise e


class NumpyLoader(Producer):
    def __init__(self, name, data_path, **kwargs):
        super().__init__(type="NPY", name=name, **kwargs)
        self._data_path = data_path
        self._data = None

    async def load_data(self):
        if self._data_path[-3:] == "npy":
            try:
                df = np.load(self._data_path)
            except Exception as e:
                raise e
        else:
            raise TypeError("Path does not lead to a numpy array")
        return df

    async def _gather_data(self):
        try:
            self._data = await self.load_data()
            self._is_done = True
            return NPYEventGather(self._data, datetime=datetime.now(), msg=DATA_GATHER_MESSAGES("GATHERED"))
        except IOError as ioe:
            return GatherEvent(msg=DATA_GATHER_MESSAGES("FAILED"), datetime=datetime.now(), data=None)
        except Exception as e:
            raise e
