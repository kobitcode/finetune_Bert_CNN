import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

from torch.optim import AdamW
from torch.utils.data import SequentialSampler, RandomSampler, Subset, TensorDataset, DataLoader
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_scheduler

from model import ModifyModel
from config import Param
from utils import flat_accuracy, format_time, get_data_loaders

device = torch.device('cuda:0')
lbs = ['positive', 'negative']

def run(args):
    dataset = load_dataset(args.data_path)
    print(dataset)
    config = AutoConfig.from_pretrained(
        args.model_path, 
        num_labels = len(lbs)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )
    df = pd.DataFrame()
    df['text'] = dataset['train']['text']
    df['label'] = dataset['train']['label']

    X_train = dataset['train']['text']
    Y_train = dataset['train']['label']

    input_ids = []
    attn_mask = []

    for sent in tqdm(X_train):
        encoded_dict = tokenizer.encode_plus(sent, 
                                            add_special_tokens = True,
                                            max_length = 128, 
                                            pad_to_max_length = True, 
                                            return_attention_mask = True, 
                                            return_tensors = "pt")
        input_ids.append(encoded_dict['input_ids'])
        attn_mask.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim = 0)
    attn_mask = torch.cat(attn_mask, dim = 0)

    Y_train = torch.tensor(Y_train)


    dataset = TensorDataset(input_ids, attn_mask, Y_train)

    total_folds = args.n_folds
    current_fold = 0 

    fold = StratifiedKFold(n_splits=args.n_folds, shuffle = True, random_state = 100000)
    training_info = []
    total_t0 = time.time()

    max_f1_score = 0

    for train_index, test_index in fold.split(df,df['label']):
        bert = AutoModel.from_pretrained(args.model_path, config = config)
        model = ModifyModel(args, config, bert).to(device)
        current_fold = current_fold+1
        train_dataloader,validation_dataloader = get_data_loaders(args, dataset,train_index,test_index)
        num_training_steps = args.epochs * len(train_dataloader)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        for epoch_i in range(0, args.epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
            print('Training...')
            t0 = time.time()
            total_train_loss = 0
            model.train()
            for step, batch in tqdm(enumerate(train_dataloader)):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()  
                loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()


            avg_train_loss = total_train_loss / len(train_dataloader) 
            training_time = format_time(time.time() - t0)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            print("")
            print("Running Validation...")

            t0 = time.time()
            model.eval()
            total_f1_score = 0
            total_precision_score = 0
            total_recall_score = 0
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():        
                    (loss, logits) = model(b_input_ids, 
                                            token_type_ids=None, 
                                            attention_mask=b_input_mask,
                                            labels=b_labels)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                total_precision_score += precision_score(np.argmax(logits,axis=1),label_ids,average='macro')
                total_recall_score += recall_score(np.argmax(logits,axis=1),label_ids,average='macro')
                total_f1_score += f1_score(np.argmax(logits,axis=1),label_ids,average='macro')


            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            avg_precision = total_precision_score / len(validation_dataloader)
            print(" Precision: {0:.2f}".format(avg_precision))

            avg_recall = total_recall_score / len(validation_dataloader)
            print(" Precision: {0:.2f}".format(avg_recall))
            
            avg_f1_score = total_f1_score / len(validation_dataloader)
            print("  F1_score: {0:.2f}".format(avg_f1_score))

            if max_f1_score < avg_f1_score:
                 max_f1_score = avg_f1_score
                 torch.save(model, "model.pt")



if __name__ == "__main__":
    param = Param()
    args = param.args
    run(args)
