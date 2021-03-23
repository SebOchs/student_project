import pytorch_lightning as pl
from litT5 import LitFineT5, LitAsagFineT5, LitMultiT5
from torch.utils.data import DataLoader
import dataloading as dl

test_loader = DataLoader(dl.T5Dataset('datasets/preprocessed/asag_kn1_ua.npy'))
t5_test = LitFineT5.load_from_checkpoint("models/multi_then_fine_kn1_t5_epoch=5-my_metric=0.1450.ckpt")
trainer = pl.Trainer(gpus=1)
trainer.test(t5_test, test_dataloaders=test_loader)
print("finished testing")