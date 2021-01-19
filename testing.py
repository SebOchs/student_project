import pytorch_lightning as pl
from lit_asag_t5 import LitT5

t5_test = LitT5.load_from_checkpoint("models/kn1_t5_epoch=22-bleu=52.6571.ckpt")
trainer = pl.Trainer(
    gpus=1
)
trainer.test(t5_test)
print("finished testing")