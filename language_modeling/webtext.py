import os
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

# os.environ['HF_HOME'] = '/storage/vsub851/.cache' ## SET THIS ENVIRONMENT IF NECESSARY!
import datasets
import transformers

from tqdm import tqdm

def load_and_tokenize_webtext():
    dataset = datasets.load_dataset('Skylion007/openwebtext', split = 'train')
    
    tokenizer = transformers.BertTokenizerFast.from_pretrained(
        'bert-base-cased', 
        do_lower_case = False, 
        cache_dir = '/storage/vsub851/.cache'
    )
    
    def tokenize_function(examples):
        texts = [text.strip().replace('\n', '[SEP]').replace('<unk>', '[UNK]')
                 for text in examples['text']]
        tokenized = tokenizer(texts, add_special_tokens = False)
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched = True, 
        num_proc = 100, 
        remove_columns=['text']
    )
    return tokenized_dataset, tokenizer

def group_texts(examples, seq_len):
    concatenated = sum(examples['input_ids'], [])
    total_length = len(concatenated)
    total_length = (total_length // seq_len) * seq_len
    result = {
        'input_ids': [concatenated[i: i + seq_len] for i in range(0, total_length, seq_len)]
    }
    return result

def prepare_grouped_dataset(save_path, seq_len = 75):
    tokenized_dataset, tokenizer = load_and_tokenize_webtext()
    
    grouped_dataset = tokenized_dataset.map(
        lambda examples: group_texts(examples, seq_len),
        batched = True,
        num_proc = 24,
        remove_columns=['token_type_ids', 'attention_mask']
    )
    
    grouped_dataset = grouped_dataset.remove_columns(grouped_dataset.column_names[:-1])
    
    os.makedirs(save_path, exist_ok = True)
    grouped_dataset.save_to_disk(save_path)
    return save_path, tokenizer

class TextDataset(data.Dataset):
    def __init__(self, arrow_dataset):
        self.dataset = arrow_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        tokens = torch.tensor(example['input_ids'], dtype = torch.long)
        return {'input_ids': tokens, 'target_ids': tokens}

def make_webtext_dataloaders(batch_size, num_workers = 12, seq_len = 128, reload_dataset = False, distributed = False):
    save_path = os.path.join('rnn_scaling', 'tokenized_webtext')
    if not os.path.exists(save_path) or reload_dataset:
        save_path, tokenizer = prepare_grouped_dataset(save_path, seq_len)
    else:
        tokenizer = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-cased', 
            do_lower_case = False, 
            cache_dir = '/storage/vsub851/.cache'
        )
    
    grouped_dataset = datasets.load_from_disk(save_path)
    dataset = TextDataset(grouped_dataset)
    
    num_total = len(dataset)
    num_train = int(0.97 * num_total)
    train_dataset = data.Subset(dataset, list(range(num_train)))
    val_dataset = data.Subset(dataset, list(range(num_train, num_total)))

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle = False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = (train_sampler == None), sampler = train_sampler, pin_memory = True, prefetch_factor = 2)
    val_loader = data.DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False, sampler = val_sampler, pin_memory = True, prefetch_factor = 2)
    return train_loader, val_loader, len(tokenizer.vocab)

if __name__ == '__main__':
    train_loader, val_loader, vocab_size = make_webtext_dataloaders(batch_size = 256, seq_len = 128)
    for batch in train_loader:
        print(batch['input_ids'].shape)
        break
    print(f'Vocab size: {vocab_size}')