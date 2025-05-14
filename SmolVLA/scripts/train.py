from src.data.dataset import CustomDataset
from src.models.vla_model import VLAWithActionHead
from src.training.trainer import build_trainer
from src.utils.memory import setup_cuda

def main():
    setup_cuda()  # CUDA memory config 
    
    # Dataset Setup 
    pytorch_dataset = CustomDataset(
        generator=tf_to_pytorch_generator(flattened_dataset), 
        length=10 
    )

    
    # Model Setup 
    model = VLAWithActionHead(...)
    model = prepare_model_for_kbit_training(base_model)  # Explicitly set parameter dtype
    model = VLAWithActionHead(model, action_head)
    model = get_peft_model(model, LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # Enable on the wrapped model [[2]]
    print(model.get_nb_trainable_parameters())  # Check trainable params

    
    # Trainer Setup
    trainer = build_trainer(model, dataset, collate_fn)
    trainer.train()

if __name__ == "__main__":
    main()