import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None   
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            # save model state dict
            self.best_model_state = model.state_dict() 

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
