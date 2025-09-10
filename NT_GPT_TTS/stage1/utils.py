class Hparams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = Hparams(**v)
            self[k] = v

    def keys(self,):
        return self.__dict__.keys()

    def values(self,):
        return self.__dict__.values()

    def item(self,):
        return self.__dict__.items()

    def __len__(self,):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self,):
        return self.__dict__.__repr__()
