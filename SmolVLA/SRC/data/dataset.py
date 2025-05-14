import tensorflow as tf
from PIL import Image
import io
import numpy as np


import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
torch.cuda.empty_cache()
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  
#import os
torch.cuda.empty_cache() 
import itertools
from torch.nn import Linear, MSELoss
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from transformers import AutoProcessor, Idefics3ForConditionalGeneration, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
import flash_attn


# Combined feature description
feature_description = {
    # Step features
    "steps/action": tf.io.FixedLenSequenceFeature([7], tf.float32, allow_missing=True),
    "steps/language_embedding": tf.io.FixedLenSequenceFeature([512], tf.float32, allow_missing=True),
    "steps/language_instruction": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/is_terminal": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "steps/is_last": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "steps/is_first": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "steps/reward": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    "steps/discount": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),

    # Observation features
    "steps/observation/image_0": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/observation/image_1": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/observation/image_2": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/observation/image_3": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/observation/state": tf.io.FixedLenSequenceFeature([7], tf.float32, allow_missing=True),

    # Episode metadata
    "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),
    "episode_metadata/episode_id": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_image_0": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_image_1": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_image_2": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_image_3": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_language": tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(serialized_example):
    parsed = tf.io.parse_single_example(serialized_example, feature_description)

    # Convert int64 to bool for flags
    def to_bool(tensor):
        return tf.cast(tensor, tf.bool)

    # Reconstruct nested structure
    steps = {
        "action": parsed["steps/action"],
        "language_embedding": parsed["steps/language_embedding"],
        "language_instruction": parsed["steps/language_instruction"],
        "is_terminal": to_bool(parsed["steps/is_terminal"]),
        "is_last": to_bool(parsed["steps/is_last"]),
        "is_first": to_bool(parsed["steps/is_first"]),
        "reward": parsed["steps/reward"],
        "discount": parsed["steps/discount"],
        "observation": {
            "image_0": parsed["steps/observation/image_0"],
            "image_1": parsed["steps/observation/image_1"],
            "image_2": parsed["steps/observation/image_2"],
            "image_3": parsed["steps/observation/image_3"],
            "state": parsed["steps/observation/state"],
        }
    }

    episode_metadata = {
        "file_path": parsed["episode_metadata/file_path"],
        "episode_id": parsed["episode_metadata/episode_id"],
        "has_image_0": to_bool(parsed["episode_metadata/has_image_0"]),
        "has_image_1": to_bool(parsed["episode_metadata/has_image_1"]),
        "has_image_2": to_bool(parsed["episode_metadata/has_image_2"]),
        "has_image_3": to_bool(parsed["episode_metadata/has_image_3"]),
        "has_language": to_bool(parsed["episode_metadata/has_language"]),
    }

    return {
        "steps": steps,
        "episode_metadata": episode_metadata
    }
    
    

#  Flatten Dataset into Individual Steps
def flatten_episode(episode):
    steps = episode["steps"]
    num_steps = tf.shape(steps["action"])[0]

    def get_step(i):
        return {
            "action": steps["action"][i],
            "language_instruction": steps["language_instruction"][i],
            "image_0": steps["observation"]["image_0"][i],  # Use one image per step
            "language_embedding": steps["language_embedding"][i],
            "is_terminal": steps["is_terminal"][i],
            "is_last": steps["is_last"][i],
            "is_first": steps["is_first"][i],
            "reward": steps["reward"][i],
            "discount": steps["discount"][i],
            #"observation": {
                #"image_0": steps["observation"]["image_0"][i],
                #"state": steps["observation"]["state"][i],
            #"observation": steps["observation"]["state"][i],
            #}'''
        }

    indices = tf.range(num_steps)
    return tf.data.Dataset.from_tensor_slices(indices).map(get_step)

# Transform Steps for `collate_fn`
def transform_step(step):
    # Decode image bytes to PIL Image
    image_bytes = step["image_0"].numpy()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((128, 128))
    image = torch.tensor(np.array(image), dtype=torch.uint8)  # Convert to tensor early [[3]]

    # Decode language instruction and action
    question = step["language_instruction"].numpy().decode("utf-8")
    action = step["action"].numpy()
    language_embedding = step["language_embedding"].numpy()
    #answer = f"Action: {action}"  # Format action as a text response
    #state = step["observation"]["state"].numpy().tolist()
    reward = step["reward"].numpy().item()
    discount = step["discount"].numpy().item()
    is_terminal = bool(step["is_terminal"].numpy().item())  # Convert int64 to bool
    is_last = bool(step["is_last"].numpy().item())
    is_first = bool(step["is_first"].numpy().item())



    return {
        "image": image,              # PIL Image (image_0 only)
        "question": question,
        "language_embedding": language_embedding,  # 512D vector [[6]]
        "action": action,            # 7D action vector
        #"state": state,              # 7D state vector
        "reward": reward,            # Scalar
        "discount": discount,        # Scalar
        "is_terminal": is_terminal,  # bool
        "is_last": is_last,          # bool
        "is_first": is_first,        # bool
    }
    
    
def collate_fn(examples):
    texts = []  # For tokenized messages
    images = []
    questions = []
    language_embeddings = []
    actions = []
    #states = []
    rewards = []
    discounts = []
    is_terminals = []
    is_lasts = []
    is_firsts = []

    for example in examples:
        # Collect data
        images.append([example["image"]])  # Single image (PIL.Image)
        questions.append(example["question"])
        language_embeddings.append(example["language_embedding"])  # 512D embedding
        actions.append(example["action"])  # 7D action vector
        #states.append(example["state"])  # 7D state vector
        rewards.append(example["reward"])  # Scalar
        discounts.append(example["discount"])  # Scalar
        is_terminals.append(example["is_terminal"])  # bool
        is_lasts.append(example["is_last"])  # bool
        is_firsts.append(example["is_first"])  # bool


                # --- Step 1: Create conversational messages [[1]][[8]] --

        messages = [
            {"role": "user", "content": [

                {"type": "text", "text": "What action should the robot take to"},  # System prompt [[1]]
                {"type": "image"},  # Image placeholder [[3]]
                {"type": "text", "text": example["question"]}  # Original instruction [[1]]
            ]},
            {"role": "assistant", "content": [
                {"type": "text","text": (
                    f"Action: {example['action']}\n"  # 7D action vector [[9]]
                    #f"State: {example['state']}\n"  # 7D state vector [[9]]
                    f"Reward: {example['reward']}\n"  # Scalar reward [[6]]
                    f"Discount: {example['discount']}\n"  # Discount factor [[6]]
                    f"Terminal: {example['is_terminal']}\n"  # Episode termination flag [[6]]
                    f"Last: {example['is_last']}\n"  # Last step flag [[6]]
                    f"First: {example['is_first']}"  # First step flag [[6]]
                )}]}
        ]

        # Apply chat template to generate tokenized text [[8]]
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors=None  # Return as token IDs
        )
        texts.append(text)



    # Process text and images 
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )


    # -Add structured features to batch 

    batch["actions"] = torch.tensor(np.array(actions), dtype=torch.float16)  # (batch_size, 7)
    #batch["labels"][batch["labels"] == processor.tokenizer.pad_token_id] = -100  # Mask padding [[7]]
    del texts, images, actions 
    torch.cuda.empty_cache()  # Clear GPU cache after batch construction [[2]][[3]]

    return batch 



# Load TFRecord dataset
tfrecord_path = "/home/mundus/fmustapha143/AIPROJECT3/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024" #Replace with path to your dataset
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
parsed_dataset = raw_dataset.map(parse_example)
flattened_dataset = parsed_dataset.flat_map(flatten_episode)  # Flatten episodes into steps [[5]]


from torch.utils.data import Dataset
import tensorflow as tf

# Define the generator function 
def tf_to_pytorch_generator(tf_dataset):
    for example in tf_dataset:  # Convert TF data to NumPy [[5]]
        yield transform_step(example)  # Apply your transformation [[3]]

#  Custom Dataset Class 
class CustomDataset(Dataset):
    def __init__(self, generator, length):
        self.data = list(itertools.islice(generator, length))
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]

#Create PyTorch Dataset 
pytorch_dataset = CustomDataset(
    generator=tf_to_pytorch_generator(flattened_dataset),  # Now defined [[5]]
    length=10  # Precomputed total samples (52 episodes × 28 steps) [[8]]
)
