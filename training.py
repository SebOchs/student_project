import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_asag_t5 import LitT5

checkpoint_callback = ModelCheckpoint(
    monitor='bleu',
    mode="max",
    filepath='models/kn1_t5_{epoch}-{bleu:.4f}',
    save_top_k=3
)
t5_test = LitT5(16, True)
trainer = pl.Trainer(
    gpus=2,
    num_nodes=1,
    accelerator='ddp',
    max_epochs=32,
    accumulate_grad_batches=1,
    checkpoint_callback=checkpoint_callback,
    # reload_dataloaders_every_epoch=True,
    num_sanity_val_steps=0,
    progress_bar_refresh_rate=10,
    # precision=16
)

trainer.fit(t5_test)

print("finished training")
