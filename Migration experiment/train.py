# baseline(Bart)
# import os
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import argparse
# import shutil
# import json
# from tqdm import tqdm
# from datetime import date
# from misc import MetricLogger, seed_everything, ProgressBar
# from load_kb import DataForSPARQL
# from data import DataLoader
# from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
# # from .sparql_engine import get_sparql_answer
# import torch.optim as optim
# import logging
# import time
# from lr_scheduler import get_linear_schedule_with_warmup
# from predict import validate
# from executor_rule import RuleExecutor
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
# logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
# rootLogger = logging.getLogger()
# import warnings
# warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query


# def train(args):
#     # device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     device = torch.device(args.gpu if args.use_cuda else "cpu")

#     logging.info("Create train_loader and val_loader.........")
#     vocab_json = os.path.join(args.input_dir, 'vocab.json')
#     train_pt = os.path.join(args.input_dir, 'train.pt')
#     val_pt = os.path.join(args.input_dir, 'val.pt')
#     train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
#     val_loader = DataLoader(vocab_json, val_pt, 64)  # DataLoader去data.py里面找，默认的training是False

#     vocab = train_loader.vocab
#     kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))
#     # rule_executor = RuleExecutor(vocab, os.path.join(args.input_dir, 'kb.json'))
#     rule_executor = RuleExecutor(os.path.join(args.input_dir, 'kb.json'))
#     logging.info("Create model.........")
#     config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
#     tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
#     model = model_class.from_pretrained(args.model_name_or_path)
#     model = model.to(device)
#     logging.info(model)
#     t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs    # 准备优化器和时间表(线性预热和衰减)
#     no_decay = ["bias", "LayerNorm.weight"]
#     bart_param_optimizer = list(model.named_parameters())
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay': args.weight_decay, 'lr': args.learning_rate},
#         {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
#          'lr': args.learning_rate}
#     ]
#     args.warmup_steps = int(t_total * args.warmup_proportion)  # total为147475
#     optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
#                                                 num_training_steps=t_total)
#     # Check if saved optimizer or scheduler states exist
#     if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
#             os.path.join(args.model_name_or_path, "scheduler.pt")):
#         # Load in optimizer and scheduler states
#         optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
#         scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

#     # Train!
#         logging.info("***** Running training *****")
#         logging.info("  Num examples = %d", len(train_loader.dataset))
#         logging.info("  Num Epochs = %d", args.num_train_epochs)
#         logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
#         logging.info("  Total optimization steps = %d", t_total)

#     global_step = 0
#     steps_trained_in_current_epoch = 0
#     # Check if continuing training from a checkpoint
#     if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
#         # set global_step to gobal_step of last saved checkpoint from model path
#         global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
#         epochs_trained = global_step // (len(train_loader) // args.gradient_accumulation_steps)
#         steps_trained_in_current_epoch = global_step % (len(train_loader) // args.gradient_accumulation_steps)
#         logging.info("  Continuing training from checkpoint, will skip to saved global_step")
#         logging.info("  Continuing training from epoch %d", epochs_trained)
#         logging.info("  Continuing training from global step %d", global_step)
#         logging.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
#     logging.info('Checking...')
#     logging.info("===================Dev==================")
#     validate(args, kb, model, val_loader, device, tokenizer, rule_executor)
#     tr_loss, logging_loss = 0.0, 0.0
#     model.zero_grad()
#     prefix = 25984
#     acc = 0.0
#     for _ in range(int(args.num_train_epochs)):
#         pbar = ProgressBar(n_total=len(train_loader), desc='Training')
#         for step, batch in enumerate(train_loader):
#             # Skip past any already trained steps if resuming training
#             if steps_trained_in_current_epoch > 0:
#                 steps_trained_in_current_epoch -= 1
#                 continue
#             model.train()
#             batch = tuple(t.to(device) for t in batch)
#             pad_token_id = tokenizer.pad_token_id
#             source_ids, source_mask, y = batch[0], batch[1], batch[-2]
#             y_ids = y[:, :-1].contiguous()
#             lm_labels = y[:, 1:].clone()
#             lm_labels[y[:, 1:] == pad_token_id] = -100

#             inputs = {
#                 "input_ids": source_ids.to(device),
#                 "attention_mask": source_mask.to(device),
#                 "decoder_input_ids": y_ids.to(device),
#                 "labels": lm_labels.to(device),  # 因为报错，后来发现将lm_labels改成labels就可以
#             }
#             outputs = model(**inputs)
#             loss = outputs[0]
            
#             loss.backward()
#             pbar(step, {'loss': loss.item()})
#             tr_loss += loss.item()
#             if (step + 1) % args.gradient_accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#                 optimizer.step()
#                 scheduler.step()  # Update learning rate schedule
#                 model.zero_grad()
#                 global_step += 1
#         validate(args, kb, model, val_loader, device, tokenizer, rule_executor)
#         output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         model_to_save = (
#             model.module if hasattr(model, "module") else model
#         )  # Take care of distributed/parallel training
#         model_to_save.save_pretrained(output_dir)
#         torch.save(args, os.path.join(output_dir, "training_args.bin"))
#         logging.info("Saving model checkpoint to %s", output_dir)
#         tokenizer.save_vocabulary(output_dir)
#         torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
#         torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
#         logging.info("Saving optimizer and scheduler states to %s", output_dir)
#         logging.info("\n")
        
#         if 'cuda' in str(device):
#             torch.cuda.empty_cache()
#     return global_step, tr_loss / global_step


# def main():
#     parser = argparse.ArgumentParser()
#     # input and output
#     parser.add_argument('--input_dir', default='./try_cycle_generation/multi_teacher/Stage_2_3_three_teacher_selfkopl_encoder_sparql_dcs/preprocessed_data')
#     parser.add_argument('--output_dir', default='./try_cycle_generation/multi_teacher/Stage_2_3_three_teacher_selfkopl_encoder_sparql_dcs/output/checkpoint')

#     parser.add_argument('--save_dir', default='./try_cycle_generation/multi_teacher/Stage_2_3_three_teacher_selfkopl_encoder_sparql_dcs/log', help='path to save checkpoints and logs')
#     parser.add_argument('--model_name_or_path', default='./bart-base')
#     parser.add_argument('--ckpt')

#     # training parameters
#     parser.add_argument('--weight_decay', default=1e-5, type=float)
#     parser.add_argument('--batch_size', default=16, type=int)
#     parser.add_argument('--seed', type=int, default=666, help='random seed')
#     parser.add_argument('--learning_rate', default=3e-5, type = float)
#     parser.add_argument('--num_train_epochs', default=25, type = int)
#     parser.add_argument('--save_steps', default=448, type = int)
#     parser.add_argument('--logging_steps', default=448, type = int)
#     parser.add_argument('--warmup_proportion', default=0.1, type = float,
#                         help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
#                         help="Epsilon for Adam optimizer.")
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.", )
#     parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                         help="Max gradient norm.")
    
#     # validating parameters
#     # parser.add_argument('--num_return_sequences', default=1, type=int)
#     # parser.add_argument('--top_p', default=)
#     # model hyperparameters
#     parser.add_argument('--dim_hidden', default=1024, type=int)
#     parser.add_argument('--alpha', default = 1e-4, type = float)

#     # 下面3行是为了指定gpu加上去的,还有一处就是30行的代码
#     # 另外一种修改的方式就是CUDA_VISIBLE_DEVICES=1,4 python train.py
#     parser.add_argument('--use_cuda', type=bool, default=True)  
#     parser.add_argument('--gpu', type=int, default=0)
#     os.environ["CUDA_VISIBLE_DEVICES"]="6,1"

#     args = parser.parse_args()

#     if not os.path.exists(args.save_dir):  # 接下来几行都是为了生成log日志
#         os.makedirs(args.save_dir)
#     time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
#     fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.log'.format(time_)))
#     fileHandler.setFormatter(logFormatter)
#     rootLogger.addHandler(fileHandler)
#     # args display
#     for k, v in vars(args).items():  # vars(args)将args传递的参数从namespace转换为dict
#         logging.info(k+':'+str(v))  # 在控制台进行打印

#     seed_everything(666)

#     train(args)


# if __name__ == '__main__':
#     main()


# Our method(Bart)
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import numpy as np
import shutil
import json
from tqdm import tqdm
from datetime import date
from misc import MetricLogger, seed_everything, ProgressBar
from load_kb import DataForSPARQL
from data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers import BertTokenizer,BertModel
import re
# from .sparql_engine import get_sparql_answer
import torch.optim as optim
import logging
import time
from lr_scheduler import get_linear_schedule_with_warmup
from predict import validate
from executor_rule import RuleExecutor
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

p = re.compile(r'[(](.*?)[)]', re.S)


def post_process(text):
    pattern = re.compile(r'".*?"')
    nes = []
    for item in pattern.finditer(text):
        nes.append((item.group(), item.span()))
    pos = [0]
    for name, span in nes:
        pos += [span[0], span[1]]
    pos.append(len(text))
    assert len(pos) % 2 == 0
    assert len(pos) / 2 == len(nes) + 1
    chunks = [text[pos[i]: pos[i+1]] for i in range(0, len(pos), 2)]
    for i in range(len(chunks)):
        chunks[i] = chunks[i].replace('?', ' ?').replace('.', ' .')
    bingo = ''
    for i in range(len(chunks) - 1):
        bingo += chunks[i] + nes[i][0]
    bingo += chunks[-1]
    return bingo



def train(args):
    # device = torch.device(args.gpu if args.use_cuda else "cpu")

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, 64)  # DataLoader去data.py里面找，默认的training是False

    vocab = train_loader.vocab
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))
    # rule_executor = RuleExecutor(vocab, os.path.join(args.input_dir, 'kb.json'))
    rule_executor = RuleExecutor(os.path.join(args.input_dir, 'kb.json'))
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    # model = model.to(device)
    model = model.cuda()
    model = nn.DataParallel(model)

    # KD
    teacher_sparql_tokenizer = tokenizer_class.from_pretrained(args.teacher_ckpt_sparql)
    teacher_sparql_model = model_class.from_pretrained(args.teacher_ckpt_sparql)
    # teacher_sparql_model = teacher_sparql_model.to(device)
    teacher_sparql_model = teacher_sparql_model.cuda()
    teacher_sparql_model = nn.DataParallel(teacher_sparql_model)
    teacher_dcs_tokenizer = tokenizer_class.from_pretrained(args.teacher_ckpt_dcs)
    teacher_dcs_model = model_class.from_pretrained(args.teacher_ckpt_dcs)
    # teacher_dcs_model = teacher_dcs_model.to(device)
    teacher_dcs_model = teacher_dcs_model.cuda()
    teacher_dcs_model = nn.DataParallel(teacher_dcs_model)
    teacher_kopl_tokenizer = tokenizer_class.from_pretrained(args.teacher_ckpt_kopl)
    teacher_kopl_model = model_class.from_pretrained(args.teacher_ckpt_kopl)
    # teacher_kopl_model = teacher_kopl_model.to(device)
    teacher_kopl_model = teacher_kopl_model.cuda()
    teacher_kopl_model = nn.DataParallel(teacher_kopl_model)


    # 转换模型
    transfer_sparql_tokenizer = tokenizer_class.from_pretrained(args.transfer_ckpt_sparql)
    transfer_sparql_model = model_class.from_pretrained(args.transfer_ckpt_sparql)
    # transfer_sparql_model = transfer_sparql_model.to(device)
    transfer_sparql_model = transfer_sparql_model.cuda()
    transfer_sparql_model = nn.DataParallel(transfer_sparql_model)
    transfer_dcs_tokenizer = tokenizer_class.from_pretrained(args.transfer_ckpt_dcs)
    transfer_dcs_model = model_class.from_pretrained(args.transfer_ckpt_dcs)
    # transfer_dcs_model = transfer_dcs_model.to(device)
    transfer_dcs_model = transfer_dcs_model.cuda()
    transfer_dcs_model = nn.DataParallel(transfer_dcs_model)
    


    logging.info(model)
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs    # 准备优化器和时间表(线性预热和衰减)
    no_decay = ["bias", "LayerNorm.weight"]
    bart_param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)  # total为147475
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_loader.dataset))
        logging.info("  Num Epochs = %d", args.num_train_epochs)
        logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_loader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_loader) // args.gradient_accumulation_steps)
        logging.info("  Continuing training from checkpoint, will skip to saved global_step")
        logging.info("  Continuing training from epoch %d", epochs_trained)
        logging.info("  Continuing training from global step %d", global_step)
        logging.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    logging.info('Checking...')
    logging.info("===================Dev==================")
    # validate(args, kb, model, val_loader, device, tokenizer, rule_executor)
    validate(args, kb, model, val_loader, tokenizer, rule_executor)
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    prefix = 25984
    acc = 0.0
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        for step, batch in enumerate(train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.cuda() for t in batch)
            pad_token_id = tokenizer.pad_token_id
            source_ids, source_mask, y = batch[0], batch[1], batch[-2]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.cuda(),
                "attention_mask": source_mask.cuda(),
                "decoder_input_ids": y_ids.cuda(),
                "labels": lm_labels.cuda(),  # 因为报错，后来发现将lm_labels改成labels就可以
                "output_hidden_states": True,
            }

            outputs = model(**inputs)
            loss_ce = outputs[0]

            student_reps = outputs["encoder_hidden_states"]
            

            # KD
            with torch.no_grad():
                teacher_reps = teacher_kopl_model(**inputs)["encoder_hidden_states"]
            
            new_teacher_reps = teacher_reps
            new_student_reps = student_reps

            rep_loss = 0.0
            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    tmp_loss = F.mse_loss(F.normalize(student_rep, p=2, dim=1), F.normalize(teacher_rep, p=2, dim=1))
                    rep_loss += tmp_loss


            # 生成的program
            outputs_p = model.module.generate(
                input_ids=source_ids,
                max_length = 500,
            )
            outputs_program = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs_p]
            outputs_program = [post_process(output) for output in outputs_program]
            # print(outputs_program[0])
            # print(len(outputs_program))
            old_questions = [tokenizer.decode(source_id) for source_id in source_ids]
            new_questions = []
            for i in range(len(old_questions)):
                start = old_questions[i].index('>')
                end = old_questions[i].index('/')
                # old_questions[i] = old_questions[i][start + 1:end - 1]
                tmp = re.findall(p, outputs_program[i])
                tmp_pro = []
                for x in tmp:
                    if x != '':
                        tmp_pro.append(x)
                new_questions.append(old_questions[i][start + 1:end - 1] + ','.join(tmp_pro))
                # outputs_program[i] = ','.join(tmp_pro)


            new_input_ids = tokenizer.batch_encode_plus(new_questions, max_length = 512, pad_to_max_length = True, truncation = True)
            new_source_ids = torch.tensor(np.array(new_input_ids['input_ids'], dtype = np.int32)).cuda()
            new_source_mask = torch.tensor(np.array(new_input_ids['attention_mask'], dtype = np.int32)).cuda()


            
            # 生成的sparql
            outputs_s = teacher_sparql_model.module.generate(
                input_ids=new_source_ids,
                max_length = 500,
            )

            # sparql to kopl
            sparql_2_kopl = transfer_sparql_model.module.generate(
                input_ids=outputs_s,
                max_length = 500,
            )

            transfer_sparql_pad_token_id = transfer_sparql_tokenizer.pad_token_id
            transfer_sparql_y_ids= sparql_2_kopl[:, :-1].contiguous()
            transfer_sparql_lm_labels = sparql_2_kopl[:, 1:].clone()
            transfer_sparql_lm_labels[sparql_2_kopl[:, 1:] == transfer_sparql_pad_token_id] = -100

            transfer_sparql_inputs = {
                "input_ids": new_source_ids.cuda(),
                "attention_mask": new_source_mask.cuda(),
                "decoder_input_ids": transfer_sparql_y_ids.cuda(),
                "labels": transfer_sparql_lm_labels.cuda(),  # 因为报错，后来发现将lm_labels改成labels就可以
            }
            transfer_sparql_loss_ce = model(**transfer_sparql_inputs)[0]

            # 生成的dcs
            outputs_d = teacher_dcs_model.module.generate(
                input_ids=new_source_ids,
                max_length = 500,
            )

            # dcs to kopl
            dcs_2_kopl = transfer_dcs_model.module.generate(
                input_ids=outputs_d,
                max_length = 500,
            )

            transfer_dcs_pad_token_id = transfer_dcs_tokenizer.pad_token_id
            transfer_dcs_y_ids= dcs_2_kopl[:, :-1].contiguous()
            transfer_dcs_lm_labels = dcs_2_kopl[:, 1:].clone()
            transfer_dcs_lm_labels[dcs_2_kopl[:, 1:] == transfer_dcs_pad_token_id] = -100

            transfer_dcs_inputs = {
                "input_ids": new_source_ids.cuda(),
                "attention_mask": new_source_mask.cuda(),
                "decoder_input_ids": transfer_dcs_y_ids.cuda(),
                "labels": transfer_dcs_lm_labels.cuda(),  # 因为报错，后来发现将lm_labels改成labels就可以
            }
            transfer_dcs_loss_ce = model(**transfer_dcs_inputs)[0]


            outputs_kopl = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in y]
            outputs_kopl = [post_process(output) for output in outputs_kopl]

            transfer_sparql_outputs_kopl = [transfer_sparql_tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in sparql_2_kopl]
            transfer_sparql_outputs_kopl = [post_process(output) for output in transfer_sparql_outputs_kopl]
            
            transfer_dcs_outputs_kopl = [transfer_dcs_tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in dcs_2_kopl]
            transfer_dcs_outputs_kopl = [post_process(output) for output in transfer_dcs_outputs_kopl]


            sparql_kd_weight = len(set(transfer_sparql_outputs_kopl) & set(outputs_kopl)) / 16.0
            dcs_kd_weight = len(set(transfer_dcs_outputs_kopl) & set(outputs_kopl)) / 16.0
            

            # for k, v in outputs.items():
            #     print(k) 
            # 当上面的inputs = {}中不加入"output_hidden_states": True,时就只有下面四个k，加上就有下面六个key
            # loss
            # logits
            # past_key_values
            # encoder_last_hidden_state

            # loss
            # logits
            # past_key_values
            # decoder_hidden_states
            # encoder_last_hidden_state
            # encoder_hidden_states

            loss = loss_ce.mean() + args.kd_weight * rep_loss + sparql_kd_weight * transfer_sparql_loss_ce.mean() + dcs_kd_weight * transfer_dcs_loss_ce.mean()
            
            
            # break

            loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        validate(args, kb, model, val_loader, tokenizer, rule_executor)
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logging.info("Saving model checkpoint to %s", output_dir)
        tokenizer.save_vocabulary(output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logging.info("Saving optimizer and scheduler states to %s", output_dir)


        # break
        logging.info("\n")
        # if 'cuda' in str(device):
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='./try_cycle_generation/multi_teacher/Stage_2_3_three_teacher_selfkopl_encoder_sparql_dcs/preprocessed_data')
    parser.add_argument('--output_dir', default='./try_cycle_generation/multi_teacher/Stage_2_3_three_teacher_selfkopl_encoder_sparql_dcs/output_raw/checkpoint')

    parser.add_argument('--save_dir', default='./try_cycle_generation/multi_teacher/Stage_2_3_three_teacher_selfkopl_encoder_sparql_dcs/log_raw', help='path to save checkpoints and logs')
    parser.add_argument('--model_name_or_path', default='./bart-base')

    parser.add_argument('--ckpt')

    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type = float)
    parser.add_argument('--num_train_epochs', default=25, type = int)
    parser.add_argument('--save_steps', default=448, type = int)
    parser.add_argument('--logging_steps', default=448, type = int)
    parser.add_argument('--warmup_proportion', default=0.1, type = float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    
    # KD
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--kd-weight", type=float, default=0.8)
    parser.add_argument("--T", type=float, default=6.0)
    parser.add_argument("--teacher-ckpt-sparql", default='./Bart_SPARQL_50/output/checkpoint/checkpoint-best')
    parser.add_argument("--teacher-ckpt-dcs", default='./Bart_DCS_50/output/checkpoint/checkpoint-best')
    parser.add_argument("--teacher-ckpt-kopl", default='./Bart_Program_50/output_test/checkpoint/checkpoint-69161') # 这个就是baseline(Bart)模型
    parser.add_argument("--kd_rep_weight", type=float, default=0.8)

    # transfer_based
    parser.add_argument("--transfer-ckpt-sparql", default='./sparql_2_kopl_50/output/checkpoint/checkpoint-best')
    parser.add_argument("--transfer-ckpt-dcs", default='./dcs_2_kopl_50/output/checkpoint/checkpoint-best')
    

    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)

    # 下面3行是为了指定gpu加上去的,还有一处就是30行的代码
    # 另外一种修改的方式就是CUDA_VISIBLE_DEVICES=1,4 python train.py
    # parser.add_argument('--use_cuda', type=bool, default=True)  
    # parser.add_argument('--gpu', type=int, default=0)
    # 注意一下，有os.environ["CUDA_VISIBLE_DEVICES"]="6, 7"的时候，表明系统只能看到这两张卡，此时parser.add_argument('--gpu', type=int, default=0)中的数值是相对CUDA_VISIBLE_DEVICES的数值而言的，这里0表示只用到gpu 6
    #           如果没有os.environ["CUDA_VISIBLE_DEVICES"]="6, 7"，那parser.add_argument('--gpu', type=int, default=0)中的数值是针对系统而言的gpu
    # os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):  # 接下来几行都是为了生成log日志
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():  # vars(args)将args传递的参数从namespace转换为dict
        logging.info(k+':'+str(v))  # 在控制台进行打印

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()
