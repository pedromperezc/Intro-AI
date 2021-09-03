import numpy as np


class BaseMetric:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def __call__(self, *args, **kwargs):
        pass


class Precision(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        prediction = self.parameters["predictions"]
        truth = self.parameters["truth"]
        tp = np.sum(truth * prediction)
        fp = truth - prediction
        fp = len(fp[fp == -1])
        return tp / (tp + fp)


class Recall(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        prediction = self.parameters["predictions"]
        truth = self.parameters["truth"]
        fn = truth - prediction
        fn = len(fn[fn == 1])
        tp = np.sum(truth * prediction)
        return tp / (tp + fn)


class Accuracy(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        prediction = self.parameters["predictions"]
        truth = self.parameters["truth"]
        tp_tn = truth - prediction
        tp_tn = len(tp_tn[tp_tn == 0])
        fp = truth - prediction
        fp = len(fp[fp == -1])
        fn = truth - prediction
        fn = len(fn[fn == 1])
        return tp_tn / (tp_tn + fp + fn)
