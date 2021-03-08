import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from litT5 import LitFineT5, LitPreMultiT5

checkpoint_callback = ModelCheckpoint(
    monitor='my_metric',
    mode="max",
    filepath='models/multi_then_fine_kn1_t5_{epoch}-{my_metric:.4f}',
    save_top_k=3
)
t5_test = LitFineT5(4, model='multi_kn1_t5_epoch=3-my_metric=0.1030.ckpt')
trainer = pl.Trainer(
    gpus=2,
    num_nodes=1,
    accelerator='ddp',
    max_epochs=64,
    accumulate_grad_batches=4,
    checkpoint_callback=checkpoint_callback,
    reload_dataloaders_every_epoch=True,
    num_sanity_val_steps=0,
    progress_bar_refresh_rate=1000,
    # precision=16
)

trainer.fit(t5_test)

print("finished training")