from transformers import TrainingArguments

def get_training_args():
    return TrainingArguments(
    num_train_epochs=1,  # Fewer epochs for initial testing [[2]]
    per_device_train_batch_size=1,  # Match your DataLoader batch_size [[6]]
    gradient_accumulation_steps=2,  # Compensate for smaller batches [[2]]
    torch_compile=False,  # Disable until stable
    tf32=False,  # Disable TensorFloat32
    #report_to="none",  # Reduce logging overhead
    dataloader_pin_memory=False,  # Reduce CPU-GPU transfer
    warmup_steps=50,
    learning_rate=1e-4,  # Lower LR for fine-tuning stability [[2]]
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    optim="paged_adamw_8bit",  # For 8-bit optimization (memory-efficient) [[6]]
    fp16=True,  # Use bfloat16 if supported (better than fp16 for some hardware) [[6]]
    output_dir=f"./SmolVLA",  # Save checkpoints here [[2]]
    hub_model_id=f"SmolVLA",  # Push to Hugging Face Hub [[2]]
    report_to="tensorboard",  # Logging integration [[2]]
    remove_unused_columns=False,  # Preserve custom features (e.g., `language_embeddings`) [[7]]
    gradient_checkpointing=True,  # Reduce memory usage [[2]]
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=0,  # Match `num_workers` in DataLoader [[8]]
    label_names=["actions"],  # Align with `collate_fn` output [[7]]
    max_steps=2,
)  