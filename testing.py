import pytorch_lightning as pl
from lit_asag_t5 import LitT5

t5_test = LitT5().load_from_checkpoint("models/asag/epoch=12-val_macro=0.8011.ckpt")
trainer = pl.Trainer(
    gpus=2,
    num_nodes=1,
    distributed_backend='ddp'
)
trainer.test(t5_test)
print("finished testing")