import numpy as np

class SalaryClipWrapper:
    def __init__(self, model, clip_max=20):
        self.model = model
        self.clip_max = clip_max

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, None, self.clip_max)