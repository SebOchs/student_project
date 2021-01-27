import pytorch_lightning as pl
from litT5 import LitFineT5, LitAsagFineT5

t5_test = LitAsagFineT5.load_from_checkpoint("models/asag_kn1_t5_epoch=5-my_metric=0.3744.ckpt")
trainer = pl.Trainer(gpus=1)
trainer.test(t5_test)
print("finished testing")