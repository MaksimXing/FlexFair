from FedBasicFunc.Data.DataCVC import *
from FedBasicFunc.Data.DataKVA import *

def bce_dice(pred, mask):
    ce_loss = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + mask.sum(dim=(1, 2))
    dice_loss = 1 - (2 * inter / (union + 1)).mean()
    return ce_loss, dice_loss