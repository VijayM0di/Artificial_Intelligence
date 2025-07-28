import json
import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.float32):
            return str(o)
        return super(CustomJSONEncoder, self).default(o)
