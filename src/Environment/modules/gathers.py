from src.Environment.abstract.data_handlers import DataGather
from src.Environment.abstract.events import GatherEvent, DATA_GATHER_MESSAGES
from src.Environment.modules.events import CSVEventGather
import pandas as pd
from datetime import datetime, timedelta
from src.Data.data import tickers_df


class DataGatherCSV(DataGather):
    def __init__(self, name, data_path, **kwargs):
        super(DataGather, self).__init__(type="CSV", name=name, **kwargs)
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
            self.data = await self.load_data()
            self.is_done = True
            return CSVEventGather(self._data, datetime=datetime.now(), msg=DATA_GATHER_MESSAGES("GATHERED"))
        except IOError as ioe:
            return GatherEvent(msg=DATA_GATHER_MESSAGES("FAILED"), datetime=datetime.now(), data=None)
        except Exception as e:
            raise e
