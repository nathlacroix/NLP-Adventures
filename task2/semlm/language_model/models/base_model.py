class Mode:
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class BaseModel():
    def build_model(self, data, pad_ind, mode, **config):
        raise NotImplementedError
