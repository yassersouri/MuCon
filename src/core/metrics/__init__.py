class Metric:
    def add(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        result = self.add(*args, **kwargs)

        return result

    def summary(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
