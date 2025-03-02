import os
import random
import torch
import pandas as pd
import transformers
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM
import copy
import json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp, hmean
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
device = torch.device('cuda')


# init idk answers
idk_answer_list = [

]


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def interleave(a, b, size):
    assert len(a) == len(b)
    assert size > 0
    c = []
    for i in range(0, len(a), size):
        c.extend(a[i:i+size])
        c.extend(b[i:i+size])
    return c


def convert_raw_data_to_model_format(data, tokenizer, max_length, loss_type):
    question = data['input']
    answer = data['output']
    full_text = question + answer
    encoded = tokenizer(full_text, add_special_tokens=True, max_length=max_length, truncation=True)
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
    if len(label) > max_length:
        label = label[:max_length]
    return {
        'input_ids': torch.tensor(pad_input_ids),
        'labels': torch.tensor(label),
        'attention_mask': torch.tensor(pad_attention_mask),
        "loss_type": loss_type,
    }


class TaskDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, loss_type):
        self.data_ = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_type = loss_type

    def __len__(self):
        return len(self.data_)

    def __getitem__(self, idx):
        # print(idx)
        data_list = [convert_raw_data_to_model_format(self.data_[idx_], self.tokenizer, self.max_length, self.loss_type) for idx_ in idx]
        data_dict = {
            'input_ids': torch.stack([item['input_ids'] for item in data_list]),
            'labels': torch.stack([item['labels'] for item in data_list]),
            'attention_mask': torch.stack([item['attention_mask'] for item in data_list]),
            'loss_type': [item['loss_type'] for item in data_list],
        }

        return data_dict


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.task_batches = kwargs.pop('task_batches')
        super(CustomTrainer, self).__init__(*args, **kwargs)

    def get_train_dataloader(self):
        while True:
            try:
                task_batch = next(task_batches)
                yield task_batch
            except StopIteration:
                task_batches = iter(self.task_batches)
            except TypeError as te:
                print(te)
                task_batches = iter(self.task_batches)
            except UnboundLocalError as ule:
                print(ule)
                task_batches = iter(self.task_batches)

    def compute_loss(self, model, inputs, return_outputs=False):

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        loss_type = inputs["loss_type"][0]
        # print(inputs["loss_type"])
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        if loss_type == 'grad_ascent':
            loss = -1 * loss
        elif loss_type == 'eul':
            loss = 1 / loss
        elif loss_type == 'eul_seq':
            loss = (1 / loss) ** 2
        return loss


def unlearn(forget_dataset_path, retain_dataset_path, ft_model_path, output_model_path):


    # get hard_device info
    num_devices = int(os.environ.get('WORLD_SIZE', 1))

    # init h-para
    max_length = 10000
    batch_size = 32
    num_epochs = 5
    lr = 1e-4
    gradient_accumulation_steps = 1
    weight_decay = 0.01
    lora_para = {
      'r': 8,
      'alpha': 32,
      'dropout': 0.05
    }
    # LOCAL_RANK = None
    flash_attention2_switch = True
    torch_dtype = torch.bfloat16
    gradient_checkpointing = True

    # set deepspeed config
    deepspeed_cfg = {
        "zero_optimization": {
            "stage": 0,
            "offload_optimizer": {
                "device": "none",
                "pin_memory": True
            },
            "offload_param": {
                "device": "none",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "train_batch_size": batch_size*gradient_accumulation_steps*num_devices,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": {
            "enabled": True
        }
    }

    # load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
        tokenizer.pad_token = tokenizer.eos_token
    except BaseException as e:
        tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-7B-0724-Instruct-hf')
        tokenizer.pad_token = tokenizer.eos_token

    # load_data
    if forget_dataset_path.endswith('parquet'):
        forget_train_list = pd.read_parquet(forget_dataset_path, engine='pyarrow').to_dict(orient='records')
    elif forget_dataset_path.endswith('jsonl'):
        forget_train_list = [json.loads(line) for line in open(forget_dataset_path, "r", encoding="utf-8").readlines()]
    if retain_dataset_path.endswith('parquet'):
        retain_train_list = pd.read_parquet(retain_dataset_path, engine='pyarrow').to_dict(orient='records')
    elif retain_dataset_path.endswith('jsonl'):
        retain_train_list = [json.loads(line) for line in open(retain_dataset_path, "r", encoding="utf-8").readlines()]
    idk_train_list = list()     # idk_data
    for line in forget_train_list:
        idk_dict = copy.deepcopy(line)
        idk_dict['split'] = 'idk'
        idk_dict['output'] = 'idk'
        random_integer = int(random.randint(0, len(idk_answer_list)-1))
        idk_dict['output'] = idk_answer_list[random_integer]
        idk_train_list.append(idk_dict)

    # process data
    def batchify(dataset, batch_size):
        """将数据集按batch_size划分成多个batch"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # print(dataloader)
        return list(dataloader)


    retain_dataset = TaskDataset(data=retain_train_list, tokenizer=tokenizer, max_length=300, loss_type="ft")
    idk_dataset = TaskDataset(data=idk_train_list, tokenizer=tokenizer, max_length=300, loss_type="ft")
    forget_dataset = TaskDataset(data=forget_train_list, tokenizer=tokenizer, max_length=300, loss_type="eul_seq")

    retain_batches = batchify(retain_dataset, batch_size)
    idk_batches = batchify(idk_dataset, batch_size)
    forget_batches = batchify(forget_dataset, batch_size)

    # merge all data
    task_batches = list()
    all_data_set = retain_batches + idk_batches + forget_batches

    # generate train_data based on input data and num_epochs
    for epoch in range(num_epochs):
        random.shuffle(all_data_set)
        random.shuffle(all_data_set)
        task_batches += copy.deepcopy(all_data_set)

    # check out path
    Path(output_model_path).mkdir(parents=True, exist_ok=True)

    max_steps = int(num_epochs*(len(forget_train_list)+len(forget_train_list)+len(retain_train_list)))//(batch_size*gradient_accumulation_steps*num_devices)

    # init train args
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, max_steps//num_epochs),
        max_steps=max_steps,
        learning_rate=lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        logging_dir=f'{output_model_path}/logs',
        output_dir=output_model_path,
        optim="paged_adamw_32bit",
        save_steps=max_steps,
        save_only_model=True,
        ddp_find_unused_parameters= False,
        remove_unused_columns=False,
        deepspeed=deepspeed_cfg,
        weight_decay = weight_decay,
        seed = 42,
        report_to="none"
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(input_model_path, use_flash_attention_2=flash_attention2_switch, torch_dtype=torch_dtype, trust_remote_code = False).to(device)

    if gradient_checkpointing is True:
        model.gradient_checkpointing_enable()
    model.generation_config.do_sample = True
    if lora_para['r'] != 0:
        config = LoraConfig(
            r=lora_para['r'],
            lora_alpha=lora_para['alpha'],
            target_modules=find_all_linear_names(model),
            lora_dropout=lora_para['dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.enable_input_require_grads()

    # init trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        task_batches=task_batches,
        train_dataset=retain_dataset,
        eval_dataset=retain_dataset,
    )

    # train model
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    if lora_para['r'] != 0:
        print("merged")
        model = model.merge_and_unload()

    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    return model
    
if __name__ == "__main__":
    forget_data_path = sys.argv[1]
    retain_data_path = sys.argv[2]
    input_model_path = sys.argv[3]
    output_model_path = sys.argv[4]
    model = unlearn(forget_data_path, retain_data_path, input_model_path, output_model_path)
