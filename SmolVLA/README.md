In this project we finetunned the SmolVLM into a Vision Language Action (VLA).

We build on the finetuning code released by Hugging face (Makers of SmolVLM) for finetuning SmolVLM. We adjusted this code to finetune the VLM into a VLA and also work with the the BrigdeV2 datset structure.


First you need to set up a virtual Enviroment

pip install -q accelerate datasets peft bitsandbytes tensorboard

pip install -q flash-attn --no-build-isolation

