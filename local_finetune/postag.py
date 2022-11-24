import pandas as pd
import argparse
import numpy as np
import transformers
import math
import torch
import os
import random
import logging
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from typing import Any, Dict, Tuple, Union
import collections
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)



DistributedConfig = collections.namedtuple(
    'DistributedConfig',
    ['device', 'local_rank', 'world_size', 'is_main_process'],
)

def distributed_setup(local_rank):
    """Sets up distributed training."""
    world_size = os.getenv('WORLD_SIZE')
    if world_size is None:
        world_size = -1
    if local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', local_rank)
        if world_size != -1:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=local_rank,
                world_size=world_size,
            )
    is_main_process = local_rank in [-1, 0] or world_size == -1
    logger.info('Rank: %s - World Size: %s', local_rank, world_size)
    return DistributedConfig(device, local_rank, world_size, is_main_process)

class Metric:
    score_key = None
    keys = None

    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, labels, predictions):
        raise NotImplementedError

    def reduce(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class Accuracy(Metric):
    score_key = 'accuracy'
    keys = ['accuracy']

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, labels, predictions):
        labels = labels.view(-1,1)
        predictions = predictions.view(-1,1)
        self.correct += labels.eq(predictions).sum()
        self.total += labels.size(0)

    def reduce(self):
        torch.distributed.reduce(self.correct, 0)
        torch.distributed.reduce(self.total, 0)

    def get(self):
        return {'accuracy': self.correct.item() / (self.total + 1e-13)}



class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return

class Trainer:
    def __init__(
        self,
        args: Dict[str, Any],
        config: transformers.PretrainedConfig,
        tokenizer: transformers.PreTrainedTokenizer,
        distributed_config: DistributedConfig,
        writer: Union[SummaryWriter, NullWriter],
    ) -> None:
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.distributed_config = distributed_config
        self.writer = writer

    def train(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
    ) -> Tuple[torch.nn.Module, float]:
        raise NotImplementedError

    def test(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
    ) -> float:
        raise NotImplementedError


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def to_device(data, device):
    """Places all tensors in data on the given device."""
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [to_device(x, device) for x in data]
    if isinstance(data, torch.Tensor):
        return data.to(device)
    raise ValueError(
        'Could not place on device: `data` should be a tensor or dictionary/list '
        'containing tensors.'
    )


METRICS = {
    'accuracy': Accuracy,
}

def get_optimizer(model, args):
  
    # Default optimizer and kwargs
    optimizer = transformers.AdamW
    kwargs = {
        'weight_decay':1e-2,
        'eps': 1e-8,
    }

    # Finetune all by default
    return optimizer(
        model.parameters(),
        lr=args['lr'],
        **kwargs
        )


def collate_batch(batch):
   
  input_ids_list,attention_mask_list, targeted_list, label_list = [],[],[],[]
   
  for (_text,_targeted,_label) in batch:
    input_ids_list.append(_text['input_ids'])
    attention_mask_list.append(_text['attention_mask'])
    targeted_list.append(_targeted)
    label_list.append(_label)
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attention_mask=pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    targeted_list = pad_sequence(targeted_list, batch_first=True, padding_value=0)
    label_list = pad_sequence(label_list, batch_first=True, padding_value=100)
    
   
    return {'input_ids':input_ids,'attention_mask':attention_mask}, targeted_list, label_list


class POSDataset(Dataset):
    def __init__(self, tokenizer,dataset):
        super(POSDataset, self).__init__()
        self.dataset = dataset
        self.tokenizer=tokenizer
        self.target=[it for item in dataset for it in item['pos_tags']]
     
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        sentence = self.dataset[index]['tokens']
        encoded = [self.tokenizer.encode(token,add_special_tokens=False) for token in sentence]
        encoded_ = [it for item in encoded for it in item]

        targeted =[]
        for item in encoded:
            if len(item)>1:
                targeted+=(len(item)-1)*[0]
                targeted+=[1]
            elif len(item)==1:
                targeted+=[1]
        

        model_inputs={'input_ids':torch.tensor(encoded_, dtype=torch.long),
                      'attention_mask':torch.tensor([1]*len(encoded_), dtype=torch.long)}
        targeted_ = torch.tensor(targeted)
        label = torch.tensor(self.dataset[index]['pos_tags'], dtype=torch.long)
        
        assert len(label)==sum(targeted)
        assert len(encoded_)==len(targeted)
        return [model_inputs, targeted_, label]

def create_dataloader(args, tokenizer):

    dataset = load_dataset("conll2003")
    train =dataset['train']
    dev =  dataset['validation']
    test = dataset['test']
    print(len(train))
    args['train_size']=len(train)
    print(len(dev))
    print(len(test))
    train_= POSDataset(tokenizer, train)
    trainloader=DataLoader(dataset=train_,batch_size=args['bsz'],collate_fn=collate_batch,shuffle=True)
    dev_= POSDataset(tokenizer, dev)
    devloader=DataLoader(dataset=dev_,batch_size=args['bsz'],collate_fn=collate_batch,shuffle=True)
    test_= POSDataset(tokenizer, test)
    testloader=DataLoader(dataset=test_,batch_size=args['bsz'],collate_fn=collate_batch)

    return trainloader, devloader,testloader

 
class FTModel(nn.Module):
    def __init__(self, base_model, vocab_size=47):
        super().__init__()
        self.base_model = base_model

        self.fc = nn.Linear(768, vocab_size)

    def forward(self, model_inputs,targeted, labels=None):
        last_hidden_states = self.base_model(**model_inputs)[0]
        logits = self.fc(last_hidden_states) #(32,max_len,47)

        logits_active = torch.stack([logits[i][j] for i in range(len(logits)) for j in range(len(logits[i])) if int(targeted[i][j])==1])
        
        if labels!=None:
            labels = labels.view(-1)
            assert len(logits_active)==len(labels)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits_active, labels)
        
            return loss, logits_active
        else:
            return logits_active
 
class FinetuneTrainer(Trainer):
    def train(self, train_loader, dev_loader):

        # TODO(rloganiv): This is lazy.
        args = self.args

        if not os.path.exists(args['ckpt_dir']):
            os.makedirs(args['ckpt_dir'])
        # Setup model
        logger.info('Initializing model.')
        base_model = transformers.AutoModel.from_pretrained(args['model_name'],config=self.config)
        model = FTModel(base_model)
        model.to(self.distributed_config.device)

        # Restore existing checkpoint if available.
        ckpt_path = os.path.join(args['ckpt_dir'], 'pytorch_model.bin')
        if os.path.exists(ckpt_path) and not args['force_overwrite']:
            logger.info('Restoring checkpoint.')
            state_dict = torch.load(ckpt_path, map_location=self.distributed_config.device)
            model.load_state_dict(state_dict)

        # Setup optimizer
        optimizer = get_optimizer(model, args)
        total_steps = args['train_size'] * args['epochs'] / (args['bsz'] * args['accumulation_steps'])
        warmup_steps = 0.1 * total_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
 
        # TODO(ewallace): The count for partial will be inaccurate since we count *all* of the LM head
        # params, whereas we are actually only updating the few that correspond to the label token names.
        total = 0
        for param_group in optimizer.param_groups:
             for tensor in param_group['params']:
                 total += tensor.numel()
        
        logger.info(f'Updating {total} / {sum(p.numel() for p in model.parameters())} params.')

        if self.distributed_config.world_size != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args['local_rank']],
            )
        metric = METRICS[args['evaluation_metric']]()

        best_score = -float('inf')
        best_metric_dict = {}
        if not args['skip_train']:
            for epoch in range(args['epochs']):
                logger.info(f'Epoch: {epoch}')
                logger.info('Training...')
                if not args['disable_dropout']:
                    model.train()
                else:
                    model.eval()
                if self.distributed_config.is_main_process and not args['quiet']:
                    iter_ = tqdm(train_loader)
                else:
                    iter_ = train_loader
                total_loss = torch.tensor(0.0, device=self.distributed_config.device)
                denom = torch.tensor(0.0, device=self.distributed_config.device)
                metric.reset()

                optimizer.zero_grad()
                for i, (model_inputs, targeted, labels) in enumerate(iter_):
                    model_inputs = to_device(model_inputs, self.distributed_config.device)
                    targeted = to_device(targeted, self.distributed_config.device)
                    labels = to_device(labels, self.distributed_config.device)
                    loss, logits = model(model_inputs,targeted,labels)         
                    preds = logits.argmax(dim=-1)
                    metric.update(labels, preds)
                    loss /= args['accumulation_steps']
                    loss.backward()
                    if (i % args['accumulation_steps']) == (args['accumulation_steps'] - 1):
                        if args['clip'] is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    batch_size = 1.0 if args['evaluation_strategy'] == 'multiple-choice' else labels.size(0)
                    total_loss += loss.detach() * batch_size
                    denom += batch_size

                    # NOTE: This loss/accuracy is only on the subset of training data
                    # in the main process.

                    if self.distributed_config.is_main_process and not args['quiet']:
                        metric_dict = metric.get()
                        metric_string = ' '.join(f'{k}: {v:0.4f}' for k, v in metric_dict.items())
                        iter_.set_description(
                            f'loss: {total_loss / (denom + 1e-13): 0.4f}, ' +
                            metric_string
                        )

                if self.distributed_config.world_size != -1:
                    torch.distributed.reduce(total_loss, 0)
                    torch.distributed.reduce(denom, 0)
                    metric.reduce()
                if self.distributed_config.is_main_process:
                    self.writer.add_scalar('Loss/train', (total_loss / (denom + 1e-13)).item(), epoch)
                    metric_dict = metric.get()
                    for key, value in metric_dict.items():
                        self.writer.add_scalar(f'{key.capitalize()}/train', value, epoch)

                if not args['skip_eval']:
                    logger.info('Evaluating...')
                    model.eval()
                    total_loss = torch.tensor(0.0, device=self.distributed_config.device)
                    denom = torch.tensor(0.0, device=self.distributed_config.device)
                    metric.reset()

                    if self.distributed_config.is_main_process and not args['quiet']:
                        iter_ = tqdm(dev_loader)
                    else:
                        iter_ = dev_loader
                    with torch.no_grad():
                        for model_inputs, targeted, labels in iter_:

                            model_inputs = to_device(model_inputs, self.distributed_config.device)
                            targeted = to_device(targeted, self.distributed_config.device)
                            labels = to_device(labels, self.distributed_config.device)
                            loss, logits = model(model_inputs,targeted,labels) 
            
                            preds = logits.argmax(dim=-1)
                            metric.update(labels, preds)
                            batch_size = 1.0 if args['evaluation_strategy'] == 'multiple-choice' else labels.size(0)
                            total_loss += loss.detach() * batch_size
                            denom += batch_size

                            if self.distributed_config.world_size != -1:
                                torch.distributed.reduce(total_loss, 0)
                                torch.distributed.reduce(denom, 0)
                                metric.reduce()
                            if self.distributed_config.is_main_process and not args['quiet']:
                                metric_dict = metric.get()
                                metric_string = ' '.join(f'{k}: {v:0.4f}' for k, v in metric_dict.items())
                                iter_.set_description(
                                    f'loss: {total_loss / (denom + 1e-13): 0.4f}, ' +
                                    metric_string
                                )
                    if self.distributed_config.is_main_process:
                        # !!!
                        self.writer.add_scalar('Loss/dev', (total_loss / (denom + 1e-13)).item(), epoch)
                        metric_dict = metric.get()
                        for key, value in metric_dict.items():
                            self.writer.add_scalar(f'{key.capitalize()}/dev', value, epoch)


                        score = metric_dict[metric.score_key]
                        if score > best_score:
                            logger.info('Best performance so far.')
                            best_score = score
                            best_metric_dict = metric_dict
                            if self.distributed_config.world_size != -1:
                                model_to_save = model.module
                            else:
                                model_to_save = model

                            if self.distributed_config.is_main_process:
                                state_dict = model_to_save.state_dict()
                                torch.save(state_dict, ckpt_path)
                            self.tokenizer.save_pretrained(args['ckpt_dir'])
                            self.config.save_pretrained(args['ckpt_dir'])

            if os.path.exists(ckpt_path) and not args['skip_eval']:
                logger.info('Restoring checkpoint.')
                if self.distributed_config.world_size != -1:
                    model_to_load = model.module
                else:
                    model_to_load = model
                state_dict = torch.load(ckpt_path, map_location=self.distributed_config.device)
                model_to_load.load_state_dict(state_dict)
  

        return model, best_metric_dict

    def test(self, model, test_loader):

        # TODO(rloganiv): This is lazy.
        args = self.args

        if not args['skip_test']:
            ckpt_path = os.path.join(args['ckpt_dir'], 'pytorch_model.bin')
            metric = METRICS[args['evaluation_metric']](
            )
            output_fname = os.path.join(args['ckpt_dir'], 'predictions')
            model.eval()

            with torch.no_grad(), open(output_fname, 'w') as f:
                for model_inputs, targeted, labels in test_loader:
                    model_inputs = to_device(model_inputs, self.distributed_config.device)
                    targeted = to_device(targeted, self.distributed_config.device)
                    labels = to_device(labels, self.distributed_config.device)
                    logits = model(model_inputs,targeted) 
                    preds = logits.argmax(dim=-1)
                    metric.update(labels, preds)

                    # Serialize output
                    for pred in preds:
                        print(pred, file=f)

            # TODO: Update metric...
            if self.distributed_config.world_size != -1:
                metric.reduce()

            if args['tmp']:
                if os.path.exists(ckpt_path):
                    logger.info('Temporary mode enabled, deleting checkpoint.')
                    os.remove(ckpt_path)
                    if 'adapter' in args['finetune_mode']:
                        adapter_path = os.path.join(args['ckpt_dir'], 'pytorch_adapter.bin')
                        adapter_head_path = os.path.join(args['ckpt_dir'], 'pytorch_model_head.bin')
                        if os.path.exists(adapter_path):
                            os.remove(adapter_path)
                        if os.path.exists(adapter_head_path):
                            os.remove(adapter_head_path)

            metric_dict = metric.get()
            for key, value in metric_dict.items():
                self.writer.add_scalar(f'{key.capitalize()}/test', value, 0)
            metric_string = ' '.join(f'{k}: {v:0.4f}' for k, v in metric_dict.items())
            logger.info(metric_string)

            return metric_dict

def main(args):
    
    seed=args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args['evaluation_strategy'] = 'classification'  
    distributed_config = distributed_setup(args['local_rank'])
    
    if not args['debug']:
        logging.basicConfig(level=logging.INFO if distributed_config.is_main_process else logging.WARN)
        logger.info('Suppressing subprocess logging. If this is not desired enable debug mode.')
        
    if distributed_config.is_main_process:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=args['ckpt_dir'])
    else:
        writer = NullWriter()
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args['model_name'],
        add_prefix_space=True,
    )

    logger.info('Loading data.')
    print('load data')
    train_loader, dev_loader, test_loader = create_dataloader(args, tokenizer)
    if args['task']=='pos':
        num_labels=47
    config = transformers.AutoConfig.from_pretrained(args['model_name'], num_labels=num_labels)
    print('done loading dataset and model config')
    trainer = FinetuneTrainer(
        args=args,
        config=config,
        tokenizer=tokenizer,
        distributed_config=distributed_config,
        writer=writer,
    )
    model, _ = trainer.train(train_loader, dev_loader)
    trainer.test(model, test_loader=test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset & model paths
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                        help='Name or path to the underlying MLM.')
    #parser.add_argument('--train', type=str, required=True,
    #                    help='Path to the training dataset.')
    #parser.add_argument('--dev', type=str, required=True,
    #                    help='Path to the development dataset.')
    #parser.add_argument('--test', type=str, required=True,
    #                    help='Path to the test dataset.')
    parser.add_argument('--task', type=str, default='pos',
                        help='task name')

    parser.add_argument('--ckpt-dir', type=str, default='ckpt-pos/',
                        help='Path to save/load model checkpoint.')

    parser.add_argument('--evaluation-metric', type=str, default='accuracy',
                        help='Evaluation metric to use.')                            
    # Skips
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training.')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation loop during training. Good for cranking through '
                             'expensive multiple-choice experiments.')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip test.')

    # Hyperparameters
    parser.add_argument('--bsz', type=int, default=8, help='Batch size.')
    parser.add_argument('--max_length', type=int, default=64, help='max sentence length')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Number of accumulation steps.')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Global learning rate.')

    parser.add_argument('--reduction-factor', type=int, default=16,
                        help='Reduction factor if using adapters')
    parser.add_argument('--disable-dropout', action='store_true',
                        help='Disable dropout during training.')
    parser.add_argument('--clip', type=float, default=None,
                        help='Gradient clipping value.')

    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed.')

    # Additional options
    parser.add_argument('-f', '--force-overwrite', action='store_true',
                        help='Allow overwriting an existing model checkpoint.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug-level logging messages.')
    parser.add_argument('--quiet', action='store_true',
                        help='Make tqdm shut up. Useful if storing logs.')
    parser.add_argument('--tmp', action='store_true',
                        help='Remove model checkpoint after evaluation. '
                             'Useful when performing many experiments.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='For parallel/distributed training. Usually will '
                             'be set automatically.')

    args = vars(parser.parse_args())

    if args['debug']:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    print("start")
    main(args)

    
    seed=args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args['evaluation_strategy'] = 'classification'  
    distributed_config = distributed_setup(args['local_rank'])
    
    if not args['debug']:
        logging.basicConfig(level=logging.INFO if distributed_config.is_main_process else logging.WARN)
        logger.info('Suppressing subprocess logging. If this is not desired enable debug mode.')
        
    if distributed_config.is_main_process:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=args['ckpt_dir'])
    else:
        writer = NullWriter()
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args['model_name'],
        add_prefix_space=True,
    )

    logger.info('Loading data.')
    print('load data')
    train_loader, dev_loader, test_loader = create_dataloader(args, tokenizer)
    if args['task']=='pos':
        num_labels=47
    config = transformers.AutoConfig.from_pretrained(args['model_name'], num_labels=num_labels)
    print('done loading dataset and model config')
    trainer = FinetuneTrainer(
        args=args,
        config=config,
        tokenizer=tokenizer,
        distributed_config=distributed_config,
        writer=writer,
    )
    model, _ = trainer.train(train_loader, dev_loader)
    trainer.test(model, test_loader=test_loader)


