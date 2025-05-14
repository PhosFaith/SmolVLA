from transformers import Trainer
from src.models.vla_model import VLAWithActionHead

def build_trainer(model, train_dataset, data_collator):
    training_args = get_training_args()
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )