class DummyClass:
    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return self

    def __call__(self, *args, **kwargs):
        pass


dummy = DummyClass()
