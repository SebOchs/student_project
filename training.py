import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_asag_t5 import LitT5

checkpoint_callback = ModelCheckpoint(
    monitor="val_macro",
    mode="max",
    filepath='models/asag/{epoch}-{val_macro:.4f}',
    save_top_k=1
)
t5_test = LitT5()
trainer = pl.Trainer(
    gpus=1,
    #num_nodes=1,
    #distributed_backend='ddp',
    max_epochs=16,
    accumulate_grad_batches=8,
    checkpoint_callback=checkpoint_callback
)

trainer.fit(t5_test)

print("finished training")
