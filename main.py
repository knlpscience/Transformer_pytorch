import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import gc
from models.set_model import set_model
from transformers import AutoTokenizer
from utils import calculate_rouge

class Seq2SeqDataset(Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return self.src_sequences[idx], self.tgt_sequences[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in src_batch], batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in tgt_batch], batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc='Training'):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output, _ = model(src, tgt_input)
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    predictions, targets = [], []

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc='Evaluating'):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output, _ = model(src, tgt_input)
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
            
            predicted_tokens = output.argmax(dim=-1).cpu().tolist()
            target_tokens = tgt_output.view(-1).cpu().tolist()
            
            predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            
            predictions.append(predicted_text)
            targets.append(target_text)
    
    avg_loss = total_loss / len(dataloader)
    avg_rouge_1, avg_rouge_2, avg_rouge_l = calculate_rouge(predictions, targets)
    
    return avg_loss, avg_rouge_1, avg_rouge_2, avg_rouge_l

def main():
    gc.collect()
    torch.cuda.empty_cache()

    with open('/root/Transformer_torch/config.json', 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained("knlp/ko-tokenizer")

    src_vocab_size = len(tokenizer)
    tgt_vocab_size = len(tokenizer)
    max_len = config['max_len']
    d_embed = config['d_embed']
    n_layer = config['n_layer']
    d_model = config['d_model']
    h = config['h']
    d_ff = config['d_ff']
    dr_rate = config['dr_rate']
    norm_eps = config['norm_eps']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = set_model(src_vocab_size, tgt_vocab_size, device, max_len, d_embed, n_layer, d_model, h, d_ff, dr_rate, norm_eps)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # padding index 무시
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train[:1000]")

    src_sequences = []
    tgt_sequences = []
    for example in dataset:
        src_text = example['document']
        tgt_text = example['summary']

        src_seq = tokenizer.encode(src_text, truncation=True, max_length=max_len)
        tgt_seq = tokenizer.encode(tgt_text, truncation=True, max_length=max_len)

        src_sequences.append(src_seq)
        tgt_sequences.append(tgt_seq)

    custom_dataset = Seq2SeqDataset(src_sequences, tgt_sequences)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(num_epochs):
        train_loss = train(model, dataloader, criterion, optimizer, device)
        eval_loss, avg_rouge_1, avg_rouge_2, avg_rouge_l = evaluate(model, dataloader, criterion, device, tokenizer)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
        print(f"ROUGE-1: {avg_rouge_1:.4f}, ROUGE-2: {avg_rouge_2:.4f}, ROUGE-L: {avg_rouge_l:.4f}")

if __name__ == "__main__":
    main()