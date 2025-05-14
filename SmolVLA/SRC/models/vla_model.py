import torch
from torch.nn import Linear
from transformers import Idefics3ForConditionalGeneration



# Model Definition 
class VLAWithActionHead(torch.nn.Module):
    def __init__(self, base_model, action_head):
        super().__init__()
        self.base_model = base_model
        self.action_head = action_head

    def forward(self, input_ids, pixel_values, attention_mask, pixel_attention_mask=None, actions = None):
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            output_hidden_states=True
        )
        #print("Model outputs:", outputs)
        last_hidden_state = outputs.hidden_states[-1]
        action_logits = self.action_head(last_hidden_state[:, -1, :])  # Use final hidden state

        # Compute loss with labels 
        loss = None
        if actions is not None:
            loss_fct = torch.nn.MSELoss()  # For action regression
            loss = loss_fct(action_logits, actions)

        return {"loss": loss, "logits": action_logits}

    # Forward gradient checkpointing calls to the base model
    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.base_model.gradient_checkpointing_enable(*args, **kwargs)
