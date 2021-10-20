import argparse
from datetime import datetime
import json
import os
import time
from transformers import GPT2LMHeadModel,AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tnrange, tqdm
from dataset import GPT21024Dataset 
from utils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, top_k_top_p_filtering

def evaluate(args, model, eval_dataset, ignore_index, epoch, global_step=None):
    """ Returns perplexity score on validation dataset.
        Args:
            args: dict that contains all the necessary information passed by user while training
            model: finetuned gpt/gpt2 model
            eval_dataset: GPT21024Dataset object for validation data
            global_step: no. of times gradients have backpropagated
            ignore_index: token not considered in loss calculation
    """
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    eval_output_dir = args.output_dir

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = batch['article'].clone().detach().to(args.device), batch['article'].clone().detach().to(args.device)
        
        with torch.no_grad():
            logits = model(inputs)[0]
            idx = batch['sum_idx'].item() # index of separator token
            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx+1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }
    print("perplexity:", perplexity.item(), end='\n\n')

    if global_step:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            for key in sorted(result.keys()):
                f.write('\n\n')
                f.write("time = %s, %s = %s, epoch = %s\n" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(epoch)))
    return result

def train(args, model, tokenizer, train_dataset, valid_dataset, ignore_index):
    """ Trains GPT2 model and logs necessary details.
        Args:
            args: dict that contains all the necessary information passed by user while training
            model: finetuned gpt/gpt2 model
            tokenizer: GPT/GPT2 tokenizer
            train_dataset: GPT21024Dataset object for training data
            ignore_index: token not considered in loss calculation
    """

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    if not os.path.exists('./logs'):
        os.mkdir('./logs')
        
    writer = SummaryWriter('./logs')
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset,sampler=train_sampler,batch_size=args.batch_size,num_workers=args.num_workers)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(),lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,100,80000)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = tnrange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            #inputs, labels = torch.tensor(batch['article']), torch.tensor(batch['article'])
            inputs, labels = batch['article'].clone().detach(), batch['article'].clone().detach()
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            logits = model(inputs)[0]
            idx = batch['sum_idx'].item() # index of separator token
            # only consider loss on reference summary just like seq2seq models
            # print(1, logits)
            # print(1, logits.size())
            shift_logits = logits[..., idx:-1, :].contiguous()
            # print(2, shift_logits)
            # print(2, shift_logits.size())
            # print(3, labels)
            # print(3, labels.size())
            shift_labels = labels[..., idx+1:].contiguous()
            # print(4, shift_labels)
            # print(4, shift_labels.size())
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('loss', (tr_loss - logging_loss)/args.gradient_accumulation_steps, global_step)
                logging_loss = tr_loss
                print("loss:", loss.item(), end='\n\n')

        print('After epoch :', epoch+1,'updates: ')
        results = evaluate(args, model, valid_dataset, ignore_index, epoch+1, global_step)
        for key, value in results.items():
            writer.add_scalar('eval_{}'.format(key), value, global_step)
        #generate_sample(valid_dataset, tokenizer, model, num=10, train=True, device=args.device)
        model_file = os.path.join(args.model_dir, 'model_trained_after_{}_epochs.bin'.format(epoch+1))
        torch.save(model.state_dict(), model_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default=0.0001, type=float, required=False, help="learning rate")
    parser.add_argument("--seed",default=42, type=int, required=False, help="seed to replicate results")
    parser.add_argument("--n_gpu",default=1, type=int, required=False, help="no of gpu available")
    parser.add_argument("--gradient_accumulation_steps",default=32, type=int, required=False, help="gradient_accumulation_steps")
    parser.add_argument("--batch_size",default=1, type=int, required=False, help="batch_size")
    parser.add_argument("--num_workers",default=2, type=int, required=False, help="num of cpus available")
    parser.add_argument("--device",default=torch.device('cuda'), required=False, help="torch.device object")
    parser.add_argument("--num_train_epochs",default=10, type=int, required=True, help="no of epochs of training")
    parser.add_argument("--output_dir",default='./output', type=str, required=False, help="path to save evaluation results")
    parser.add_argument("--model_dir",default='./weights', type=str, required=False, help="path to save trained model")
    parser.add_argument("--fp16",default=True, type=bool, required=False, help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level",default='O0', type=str, required=False, help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("--max_grad_norm",default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("--root_dir",default='./data/gpt2_1024_data', type=str, required=False, help="location of json dataset.")
    parser.add_argument("--ids_file",default='./data/ids.json', type=str, required=False, help="location of train, valid and test file indexes")
    args = parser.parse_args()

    with open(args.ids_file,'r') as f:
        js = json.load(f)
        train_size = len(js['train_ids'])
        valid_size = len(js['valid_ids'])

    train_data = GPT21024Dataset(args.root_dir,args.ids_file,mode='train',length=train_size)
    valid_data = GPT21024Dataset(args.root_dir,args.ids_file,mode='valid',length=valid_size)
    tokenizer = add_special_tokens()
    ignore_index = tokenizer.pad_token_id
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    start = time.time()
    train(args, model, tokenizer, train_data, valid_data, ignore_index)
    print('total time: ', (time.time()-start)/60, " minutes", end='\n\n')

    print('Saving trained model...')
    model_file = os.path.join(args.model_dir, 'model_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.bin'.format(args.fp16_opt_level, train_size, args.num_train_epochs))
    config_file = os.path.join(args.model_dir, 'config_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.json'.format(args.fp16_opt_level, train_size, args.num_train_epochs))
    torch.save(model.state_dict(), model_file)
    model.config.to_json_file(config_file)


if __name__ == '__main__':
	main()
