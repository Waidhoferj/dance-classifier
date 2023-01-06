import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class LabelWeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy loss that assumes each float in the final dimension is a binary probability distribution.
    Allows for the weighing of each probability distribution wrt loss.
    """
    def __init__(self, label_weights:torch.Tensor, reduction="mean"):
        super().__init__()
        self.label_weights = label_weights

        match reduction:
            case "mean":
                self.reduction = torch.mean
            case "sum":
                self.reduction = torch.sum
    
    def _log(self,x:torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(torch.log(x), -100)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = -self.label_weights * (target * self._log(input) + (1-target) * self._log(1-input))
        return self.reduction(losses)


# TODO: Code a onehot


def calculate_metrics(pred, target, threshold=0.5, prefix="", multi_label=True) -> dict[str, torch.Tensor]:
    target = target.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    params = {
            "y_true": target if multi_label else target.argmax(1) ,
            "y_pred": np.array(pred > threshold, dtype=float) if multi_label else pred.argmax(1), 
            "zero_division": 0,
            "average":"macro"
            }
    metrics= {
            'precision': precision_score(**params),
            'recall': recall_score(**params),
            'f1': f1_score(**params),
            'accuracy': accuracy_score(y_true=params["y_true"], y_pred=params["y_pred"]),
            }
    return {prefix + k: torch.tensor(v,dtype=torch.float32) for k,v in metrics.items()}

class EarlyStopping:
    def __init__(self, patience=0):
        self.patience = patience
        self.last_measure = np.inf
        self.consecutive_increase = 0
    
    def step(self, val) -> bool:
        if self.last_measure <= val:
            self.consecutive_increase +=1
        else:
            self.consecutive_increase = 0
        self.last_measure = val

        return self.patience < self.consecutive_increase