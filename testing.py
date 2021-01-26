import pytorch_lightning as pl
from litT5 import LitFineT5

t5_test = LitFineT5.load_from_checkpoint("models/kn1_t5_epoch=46-my_metric=0.2207.ckpt")
trainer = pl.Trainer(gpus=1)
trainer.test(t5_test)
print("finished testing")