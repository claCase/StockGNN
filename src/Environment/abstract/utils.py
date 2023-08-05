import json


class ToDict:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self, return_json=False):
        var2dict = {k.strip("_"): v for k, v in self.__dict__.items()}
        if return_json:
            return json.dumps(var2dict)
        else:
            return var2dict
