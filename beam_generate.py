import argparse
import json
from transformers import GPT2LMHeadModel
import torch
from dataset import GPT21024Dataset 
from utils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, top_k_top_p_filtering
                               
parser = argparse.ArgumentParser()
parser.add_argument("--device",default=torch.device('cuda'), required=False, help="torch.device object")
parser.add_argument("--root_dir",default='./data/gpt2_1024_data', type=str, required=False, help="location of json dataset.")
parser.add_argument("--ids_file",default='./data/ids.json', type=str, required=False, help="location of train, valid and test file indexes")
parser.add_argument("--model", type=str, required=True, help="trained model location")
parser.add_argument("--num", default=10, type=int, required=False, help="number of predictions")
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
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint)

generate_beam_sample(valid_data, tokenizer, model, num=args.num, length=60, beam_size=3, device=args.device)