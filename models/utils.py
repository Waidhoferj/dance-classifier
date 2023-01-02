import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class LabelWeightedBCELoss(nn.Module):
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


def calculate_metrics(pred, target, threshold=0.5, prefix="") -> dict[str, torch.Tensor]:
    target = target.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred = np.array(pred > threshold, dtype=float)
    metrics= {
            'precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'accuracy': accuracy_score(y_true=target, y_pred=pred),
            }
    if prefix != "":
        metrics = {prefix + k : v for k, v in metrics.items()}
    
    return {k: torch.tensor(v,dtype=torch.float32) for k,v in metrics.items()}
