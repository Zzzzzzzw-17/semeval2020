import argparse
import os
import pickle
import json
import torch
import time
import logging
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.functional import log_softmax
from transformers import BertTokenizer, BertForMaskedLM
from gensim import utils as gensim_utils
from smart_open import open

logger = logging.getLogger(__name__)


class PathLineSentences(object):
    """Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a directory
    in alphabetical order by filename.

    The directory must only contain files that can be read by :class:`gensim.models.word2vec.LineSentence`:
    .bz2, .gz, and text files. Any file not ending with .bz2 or .gz is assumed to be a text file.

    The format of files (either text, or compressed text files) in the path is one sentence = one line,
    with words already preprocessed and separated by whitespace.

    Warnings
    --------
    Does **not recurse** into subdirectories.

    """

    def __init__(self, source, limit=None, max_sentence_length=100000):
        """
        Parameters
        ----------
        source : str
            Path to the directory.
        limit : int or None
            Read only the first `limit` lines from each file. Read all if limit is None (the default).

        """
        self.source = source
        self.limit = limit
        self.max_sentence_length = max_sentence_length

        if os.path.isfile(self.source):
            logger.debug('single file given as source, rather than a directory of files')
            logger.debug('consider using models.word2vec.LineSentence for a single file')
            self.input_files = [self.source]  # force code compatibility with list of files
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            logger.info('reading directory %s', self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('input is neither a file nor a path')
        logger.info('files read into PathLineSentences:%s', '\n'.join(self.input_files))

    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            logger.info('reading file %s', file_name)
            with gensim_utils.file_or_filename(file_name) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = gensim_utils.to_unicode(line, encoding='utf-8').split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length


def get_context(tokenizer, token_ids, target_position, sequence_length):
    window_size = int((sequence_length - 2) / 2)

    # determine where context starts and if there are any unused context positions to the left
    if target_position - window_size >= 0:
        start = target_position - window_size
        extra_left = 0
    else:
        start = 0
        extra_left = window_size - target_position

    # determine where context ends and if there are any unused context positions to the right
    if target_position + window_size + 1 <= len(token_ids):
        end = target_position + window_size + 1
        extra_right = 0
    else:
        end = len(token_ids)
        extra_right = target_position + window_size + 1 - len(token_ids)

    # redistribute to the left the unused right context positions
    if extra_right > 0 and extra_left == 0:
        if start - extra_right >= 0:
            padding = 0
            start -= extra_right
        else:
            padding = extra_right - start
            start = 0
    # redistribute to the right the unused left context positions
    elif extra_left > 0 and extra_right == 0:
        if end + extra_left <= len(token_ids):
            padding = 0
            end += extra_left
        else:
            padding = end + extra_left - len(token_ids)
            end = len(token_ids)
    else:
        padding = extra_left + extra_right

    context_ids = token_ids[start:end]
    context_ids = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id]
    item = {'input_ids': context_ids + padding * [tokenizer.pad_token_id],
            'attention_mask': len(context_ids) * [1] + padding * [0]}

    new_target_position = target_position - start + 1

    return item, new_target_position


class ContextsDataset(torch.utils.data.Dataset):

    def __init__(self, target_forms_i2w, form2target, sentences, context_size, tokenizer, n_sentences=None):
        super(ContextsDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.context_size = context_size

        for sentence in tqdm(sentences, total=n_sentences):
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            for spos, tok_id in enumerate(token_ids):
                if tok_id in target_forms_i2w:
                    form = target_forms_i2w[tok_id]
                    model_input, pos_in_context = get_context(tokenizer, token_ids, spos, context_size)
                    self.data.append((model_input, form2target[form], pos_in_context))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        model_input, lemma, pos_in_context = self.data[index]
        model_input = {'input_ids': torch.tensor(model_input['input_ids'], dtype=torch.long).unsqueeze(0),
                       'attention_mask': torch.tensor(model_input['attention_mask'], dtype=torch.long).unsqueeze(0)}
        return model_input, lemma, pos_in_context


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def to_numpy(tensor):
    if torch.cuda.is_available():
        return tensor.detach().cpu().clone().numpy()
    else:
        return tensor.clone().numpy()


def to_list(tensor):
    if torch.cuda.is_available():
        return tensor.detach().cpu().clone().tolist()
    else:
        return tensor.clone().tolist()


def main():
    """
    Collect lexical substitutes and their probabilities.
    """
    parser = argparse.ArgumentParser(
        description='Collect lexical substitutes and their probabilities.')
    parser.add_argument(
        '--model_name', type=str, required=True,
        help='HuggingFace model name or path')
    parser.add_argument(
        '--corpus_path', type=str, required=True,
        help='Path to corpus or corpus directory (iterates through files)')
    parser.add_argument(
        '--targets_path', type=str, required=True,
        help='Path to the csv file containing target word forms.')
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Output path for pickle containing substitutes.')
    parser.add_argument(
        "--output_postfix", default="_substitutes.json.gz",
        help="Out file postfix (added to the word)")
    parser.add_argument(
        '--n_subs', type=int, default=100,
        help='The number of lexical substitutes to extract.')
    parser.add_argument(
        '--seq_len', type=int, default=128,
        help="The length of a token's entire context window.")
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='The batch size per device (GPU core / CPU).')
    parser.add_argument(
        '--ignore_decoder_bias', action='store_true',
        help="Whether to ignore the decoder's bias vector during masked word prediction")
    parser.add_argument(
        '--local_rank', type=int, default=-1,
        help='For distributed training.')
    parser.add_argument(
        '--do_lower_case', action='store_true',
        help="Setting BERT Tokenizer to uncased mode.")
    parser.add_argument(
        '--all_target_forms', action='store_true',
        help="Whether to consider all forms of a target word -- should match the tokenizer."
    )
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()
    
    if args.local_rank in [-1, 0]:
        if os.path.exists(args.output_path):
            logger.warning('Output path exists: {}'.format(args.output_path))
            assert os.path.isdir(args.output_path), 'Output path must be a directory.'
        else:
            logger.warning('Create output directory: {}'.format(args.output_path))
            os.makedirs(args.output_path)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        n_gpu,
        bool(args.local_rank != -1)
    )

    # Set seeds across modules
    set_seed(42, n_gpu)

    # Load targets
    form2target = {}
    with open(args.targets_path, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            entries = line.split(',')
            target, forms = entries[0], entries[1:]
            for form in forms:
                if args.do_lower_case:
                    form = form.lower()
                form2target[form] = target

    print('=' * 80)
    print(form2target)
    print('=' * 80)

    if args.all_target_forms:
        targets = list(form2target.keys())
    else:
        targets = list(form2target.values())

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name, never_split=targets, use_fast=False, do_lower_case=args.do_lower_case)
    model = BertForMaskedLM.from_pretrained(args.model_name, output_hidden_states=False)

    if args.ignore_decoder_bias:
        logger.warning('Ignoring bias vector for masked word prediction.')
        model.cls.predictions.decoder.bias = None

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Get sentence iterator
    sentences = PathLineSentences(args.corpus_path)

    i2w = {}
    for w in targets:
        w_id = tokenizer.encode(w, add_special_tokens=False)
        assert len(w_id) == 1
        w_id = w_id[0]
        assert w_id != tokenizer.unk_token_id
        i2w[w_id] = w

    nSentences = 0
    target_counter = {target: 0 for target in form2target.values()}
    for sentence in sentences:
        nSentences += 1
        for tok_id in tokenizer.encode(sentence, add_special_tokens=False):
            if tok_id in i2w:
                form = i2w[tok_id]
                target_counter[form2target[form]] += 1

    logger.warning('usages: %d' % (sum(list(target_counter.values()))))

    def collate(batch):
        return [
            {'input_ids': torch.cat([item[0]['input_ids'] for item in batch], dim=0),
             'attention_mask': torch.cat([item[0]['attention_mask'] for item in batch], dim=0)},
            [item[1] for item in batch],
            [item[2] for item in batch]
        ]

    # Container for lexical substitutes
    substitutes = {
        target: [{} for _ in range(target_count)]
        for (target, target_count) in target_counter.items()
    }

    # Iterate over sentences and collect representations
    nUsages = 0
    curr_idx = {target: 0 for target in target_counter}

    dataset = ContextsDataset(i2w, form2target, sentences, args.seq_len, tokenizer, nSentences)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
    iterator = tqdm(dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(iterator):
        model.eval()

        inputs, lemmas, positions = batch
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        bsz = inputs['input_ids'].shape[0]

        with torch.no_grad():
            outputs = model(**inputs)  # n_sentences, max_sent_len, vocab_size

            logits = outputs[0][np.arange(bsz), positions, :]  # n_sentences, vocab_size
            logp = log_softmax(logits, dim=-1)
            values, indices = torch.sort(logp, dim=-1, descending=True)
            values, indices = values[:, :args.n_subs], indices[:, :args.n_subs]  # n_sentences, n_substitutes

            values = to_list(values)
            indices = to_list(indices)

            for b_id in np.arange(bsz):
                lemma = lemmas[b_id]

                substitutes[lemma][curr_idx[lemma]]['candidate_words'] = tokenizer.convert_ids_to_tokens(indices[b_id])

                assert len(substitutes[lemma][curr_idx[lemma]]['candidate_words']) == len(indices[b_id])

                substitutes[lemma][curr_idx[lemma]]['candidates'] = indices[b_id]
                substitutes[lemma][curr_idx[lemma]]['logp'] = values[b_id]

                curr_idx[lemma] += 1
                nUsages += 1

    iterator.close()

    for word in substitutes:
        if len(substitutes[word]) < 1:
            logger.warning(f"No occurrences found for {word}!")
            continue
        outfile = os.path.join(args.output_path, word + args.output_postfix)
        with open(outfile, "w") as f:
            for occurrence in substitutes[word]:
                out = json.dumps(occurrence, ensure_ascii=False)
                f.write(out + "\n")
        logger.info(f"Substitutes saved to {outfile}")

    logger.warning('usages: %d' % (nUsages))
    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
