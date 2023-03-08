# baseline(bart)

import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import json
from tqdm import tqdm
from datetime import date
from misc import MetricLogger, seed_everything, ProgressBar
from data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import torch.optim as optim
import logging
import time
from lr_scheduler import get_linear_schedule_with_warmup
from predict import validate
from kopl.kopl import KoPLEngine
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

new_tokens = ['<func>', '<arg>']

def train(args):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.gpu if args.use_cuda else "cpu")

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'count_vocab.json')
    train_pt = os.path.join(args.input_dir, 'count_train.pt')
    val_pt = os.path.join(args.input_dir, 'count_val.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, 64)

    engine = KoPLEngine(json.load(open(os.path.join(args.input_dir, 'kb.json'))))
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    added_tokens_num = tokenizer.add_tokens(new_tokens, special_tokens = True)
    print('added_tokens_num:', added_tokens_num)
    if added_tokens_num > 0:
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    logging.info(model)
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bart_param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
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
    validate(model, val_loader, device, tokenizer, engine)
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        for step, batch in enumerate(train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            pad_token_id = tokenizer.pad_token_id
            source_ids, source_mask, y = batch[0], batch[1], batch[-2]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": lm_labels.to(device),
            }
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        validate(model, val_loader, device, tokenizer, engine)
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logging.info("Saving model checkpoint to %s", output_dir)
        # tokenizer.save_vocabulary(output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logging.info("Saving optimizer and scheduler states to %s", output_dir)
        logging.info("\n")
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='./Multi_Teacher_In_Questions/preprocessed_data_next')
    parser.add_argument('--output_dir', default='./Multi_Teacher_In_Questions/count_kopl/output/checkpoint')

    parser.add_argument('--save_dir', default='./Multi_Teacher_In_Questions/count_kopl/log', help='path to save checkpoints and logs')
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
    
    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)
    # parser.add_argument('--top_p', default=)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)

    parser.add_argument('--use_cuda', type=bool, default=True)  
    parser.add_argument('--gpu', type=int, default=1)
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()


# our method
# 中间循环生成 + count(encoder) + logical and multihop -> count
# import os
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import argparse
# import json
# import torch.nn.functional as F
# import numpy as np
# import re
# from tqdm import tqdm
# from datetime import date
# from misc import MetricLogger, seed_everything, ProgressBar
# from data import DataLoader
# from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
# import torch.optim as optim
# import logging
# import time
# from lr_scheduler import get_linear_schedule_with_warmup
# from predict import validate
# from kopl.kopl import KoPLEngine
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
# logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
# rootLogger = logging.getLogger()
# import warnings
# warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

# new_tokens = ['<func>', '<arg>']

# def post_process(text):
#     pattern = re.compile(r'".*?"')
#     nes = []
#     for item in pattern.finditer(text):
#         nes.append((item.group(), item.span()))
#     pos = [0]
#     for name, span in nes:
#         pos += [span[0], span[1]]
#     pos.append(len(text))
#     assert len(pos) % 2 == 0
#     assert len(pos) / 2 == len(nes) + 1
#     chunks = [text[pos[i]: pos[i+1]] for i in range(0, len(pos), 2)]
#     for i in range(len(chunks)):
#         chunks[i] = chunks[i].replace('?', ' ?').replace('.', ' .')
#     bingo = ''
#     for i in range(len(chunks) - 1):
#         bingo += chunks[i] + nes[i][0]
#     bingo += chunks[-1]
#     return bingo

# def train(args):
#     # device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # device = torch.device(args.gpu if args.use_cuda else "cpu")

#     logging.info("Create train_loader and val_loader.........")
#     vocab_json = os.path.join(args.input_dir, 'count_vocab.json')
#     train_pt = os.path.join(args.input_dir, 'count_train.pt')
#     val_pt = os.path.join(args.input_dir, 'count_val.pt')
#     train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
#     val_loader = DataLoader(vocab_json, val_pt, 64)

#     engine = KoPLEngine(json.load(open(os.path.join(args.input_dir, 'kb.json'))))
#     logging.info("Create model.........")
#     config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
#     tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
#     model = model_class.from_pretrained(args.model_name_or_path)
#     added_tokens_num = tokenizer.add_tokens(new_tokens, special_tokens = True)
#     print('added_tokens_num:', added_tokens_num)
#     if added_tokens_num > 0:
#         model.resize_token_embeddings(len(tokenizer))

#     # model = model.to(device)
#     model = model.cuda()
#     model = nn.DataParallel(model)

#     # KD
#     teacher_count_tokenizer = tokenizer_class.from_pretrained(args.teacher_ckpt_count)
#     teacher_count_model = model_class.from_pretrained(args.teacher_ckpt_count)
#     # teacher_count_model = teacher_count_model.to(device)
#     teacher_count_model = teacher_count_model.cuda()
#     teacher_count_model = nn.DataParallel(teacher_count_model)
#     teacher_logical_tokenizer = tokenizer_class.from_pretrained(args.teacher_ckpt_logical)
#     teacher_logical_model = model_class.from_pretrained(args.teacher_ckpt_logical)
#     # teacher_logical_model = teacher_logical_model.to(device)
#     teacher_logical_model = teacher_logical_model.cuda()
#     teacher_logical_model = nn.DataParallel(teacher_logical_model)
#     teacher_multihop_tokenizer = tokenizer_class.from_pretrained(args.teacher_ckpt_multihop)
#     teacher_multihop_model = model_class.from_pretrained(args.teacher_ckpt_multihop)
#     # teacher_multihop_model = teacher_multihop_model.to(device)
#     teacher_multihop_model = teacher_multihop_model.cuda()
#     teacher_multihop_model = nn.DataParallel(teacher_multihop_model)
    

#     logging.info(model)
#     t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs    # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ["bias", "LayerNorm.weight"]
#     bart_param_optimizer = list(model.named_parameters())
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay': args.weight_decay, 'lr': args.learning_rate},
#         {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
#          'lr': args.learning_rate}
#     ]
#     args.warmup_steps = int(t_total * args.warmup_proportion)
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
#     # validate(model, val_loader, device, tokenizer, engine)
#     validate(model, val_loader, tokenizer, engine)
#     tr_loss, logging_loss = 0.0, 0.0
#     model.zero_grad()
#     for _ in range(int(args.num_train_epochs)):
#         pbar = ProgressBar(n_total=len(train_loader), desc='Training')
#         for step, batch in enumerate(train_loader):
#             # Skip past any already trained steps if resuming training
#             if steps_trained_in_current_epoch > 0:
#                 steps_trained_in_current_epoch -= 1
#                 continue
#             model.train()
#             batch = tuple(t.cuda() for t in batch)
#             pad_token_id = tokenizer.pad_token_id
#             source_ids, source_mask, y = batch[0], batch[1], batch[-2]
#             y_ids = y[:, :-1].contiguous()
#             lm_labels = y[:, 1:].clone()
#             lm_labels[y[:, 1:] == pad_token_id] = -100

#             inputs = {
#                 "input_ids": source_ids.cuda(),
#                 "attention_mask": source_mask.cuda(),
#                 "decoder_input_ids": y_ids.cuda(),
#                 "labels": lm_labels.cuda(),
#                 "output_hidden_states": True,
#             }
#             outputs = model(**inputs)
#             loss_ce = outputs[0]

#             student_reps = outputs["encoder_hidden_states"]
            

#             # KD
#             with torch.no_grad():
#                 teacher_reps = teacher_count_model(**inputs)["encoder_hidden_states"]
            
#             new_teacher_reps = teacher_reps
#             new_student_reps = student_reps

#             rep_loss = 0.0
#             for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
#                     tmp_loss = F.mse_loss(F.normalize(student_rep, p=2, dim=1), F.normalize(teacher_rep, p=2, dim=1))
#                     rep_loss += tmp_loss
            

#             # 生成的program
#             outputs_p = model.module.generate(
#                 input_ids=source_ids,
#                 max_length = 500,
#             )
#             outputs_program = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs_p]
#             outputs_program = [post_process(output) for output in outputs_program]
#             print(outputs_program[0])
#             # Find <arg> human <func> Relate <arg> studied by <arg> forward <func> FilterConcept <arg> social science <func> Count
#             # break
#             # print(len(outputs_program))
#             old_questions = [tokenizer.decode(source_id) for source_id in source_ids]
#             new_questions = []
#             for i in range(len(old_questions)):
#                 start = old_questions[i].index('>')
#                 end = old_questions[i].index('/')
#                 # old_questions[i] = old_questions[i][start + 1:end - 1]
#                 tmp_pro = []
#                 for j in outputs_program[i].split("<func>"):
#                     t = j.split("<arg>")
#                     for k in range(1, len(t)):
#                         tmp_pro.append(t[k])
                
#                 new_questions.append(old_questions[i][start + 1:end - 1] + ','.join(tmp_pro))
#                 # outputs_program[i] = ','.join(tmp_pro)
            
#             new_input_ids = tokenizer.batch_encode_plus(new_questions, max_length = 512, pad_to_max_length = True, truncation = True)
#             new_source_ids = torch.tensor(np.array(new_input_ids['input_ids'], dtype = np.int32)).cuda()
#             new_source_mask = torch.tensor(np.array(new_input_ids['attention_mask'], dtype = np.int32)).cuda()


#             # (multihop)生成的kopl
#             outputs_multihop_kopl = teacher_multihop_model.module.generate(
#                 input_ids=new_source_ids,
#                 max_length = 500,
#             )

#             transfer_multihop_pad_token_id = teacher_multihop_tokenizer.pad_token_id
#             transfer_multihop_y_ids= outputs_multihop_kopl[:, :-1].contiguous()
#             transfer_multihop_lm_labels = outputs_multihop_kopl[:, 1:].clone()
#             transfer_multihop_lm_labels[outputs_multihop_kopl[:, 1:] == transfer_multihop_pad_token_id] = -100

#             transfer_multihop_inputs = {
#                 "input_ids": new_source_ids.cuda(),
#                 "attention_mask": new_source_mask.cuda(),
#                 "decoder_input_ids": transfer_multihop_y_ids.cuda(),
#                 "labels": transfer_multihop_lm_labels.cuda(),  # 因为报错，后来发现将lm_labels改成labels就可以
#             }
#             transfer_multihop_loss_ce = model(**transfer_multihop_inputs)[0]

#             # (logical)生成的kopl
#             outputs_logical_kopl = teacher_logical_model.module.generate(
#                 input_ids=new_source_ids,
#                 max_length = 500,
#             )

#             transfer_logical_pad_token_id = teacher_logical_tokenizer.pad_token_id
#             transfer_logical_y_ids= outputs_logical_kopl[:, :-1].contiguous()
#             transfer_logical_lm_labels = outputs_logical_kopl[:, 1:].clone()
#             transfer_logical_lm_labels[outputs_logical_kopl[:, 1:] == transfer_logical_pad_token_id] = -100

#             transfer_logical_inputs = {
#                 "input_ids": new_source_ids.cuda(),
#                 "attention_mask": new_source_mask.cuda(),
#                 "decoder_input_ids": transfer_logical_y_ids.cuda(),
#                 "labels": transfer_logical_lm_labels.cuda(),  # 因为报错，后来发现将lm_labels改成labels就可以
#             }
#             transfer_logical_loss_ce = model(**transfer_logical_inputs)[0]

#             outputs_kopl = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in y]
#             outputs_kopl = [post_process(output) for output in outputs_kopl]

#             transfer_multihop_outputs_kopl = [teacher_multihop_tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs_multihop_kopl]
#             transfer_multihop_outputs_kopl = [post_process(output) for output in transfer_multihop_outputs_kopl]

#             transfer_logical_outputs_kopl = [teacher_logical_tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs_logical_kopl]
#             transfer_logical_outputs_kopl = [post_process(output) for output in transfer_logical_outputs_kopl]

            
#             multihop_kd_weight = len(set(transfer_multihop_outputs_kopl) & set(outputs_kopl)) / 16.0
#             logical_kd_weight = len(set(transfer_logical_outputs_kopl) & set(outputs_kopl)) / 16.0
#             print("multihop_kd_weight->", multihop_kd_weight)
#             print("logical_kd_weight->", logical_kd_weight)

            
            
#             loss = loss_ce.mean() + args.kd_weight * rep_loss + multihop_kd_weight * transfer_multihop_loss_ce.mean() + logical_kd_weight * transfer_logical_loss_ce.mean()
            
            
#             loss.backward()
#             pbar(step, {'loss': loss.item()})
#             tr_loss += loss.item()
#             if (step + 1) % args.gradient_accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#                 optimizer.step()
#                 scheduler.step()  # Update learning rate schedule
#                 model.zero_grad()
#                 global_step += 1
#         # break
#         # validate(model, val_loader, device, tokenizer, engine)
#         validate(model, val_loader, tokenizer, engine)
#         output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         model_to_save = (
#             model.module if hasattr(model, "module") else model
#         )  # Take care of distributed/parallel training
#         model_to_save.save_pretrained(output_dir)
#         tokenizer.save_pretrained(output_dir)
#         torch.save(args, os.path.join(output_dir, "training_args.bin"))
#         logging.info("Saving model checkpoint to %s", output_dir)
#         # tokenizer.save_vocabulary(output_dir)
#         torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
#         torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
#         logging.info("Saving optimizer and scheduler states to %s", output_dir)
#         logging.info("\n")
#         # if 'cuda' in str(device):
#         torch.cuda.empty_cache()
#     return global_step, tr_loss / global_step


# def main():
#     parser = argparse.ArgumentParser()
#     # input and output
#     parser.add_argument('--input_dir', default='./Multi_Teacher_In_Questions/preprocessed_data_next')
#     parser.add_argument('--output_dir', default='./Multi_Teacher_In_Questions/circle_count_encoder_logical_multihop_guide_count/output/checkpoint')

#     parser.add_argument('--save_dir', default='./Multi_Teacher_In_Questions/circle_count_encoder_logical_multihop_guide_count/log', help='path to save checkpoints and logs')
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

    
#     # KD
#     parser.add_argument("--ce-weight", type=float, default=0.2)
#     parser.add_argument("--kd-weight", type=float, default=0.8)
#     parser.add_argument("--T", type=float, default=6.0)
#     parser.add_argument("--teacher-ckpt-count", default='./Multi_Teacher_In_Questions/count_kopl/output/checkpoint/checkpoint-17125')
#     parser.add_argument("--teacher-ckpt-logical", default='./Multi_Teacher_In_Questions/logical_kopl/output/checkpoint/checkpoint-38134')
#     parser.add_argument("--teacher-ckpt-multihop", default='./Multi_Teacher_In_Questions/multihop_kopl/output/checkpoint/checkpoint-108500')
    
#     parser.add_argument("--kd_rep_weight", type=float, default=0.8)

    

#     # model hyperparameters
#     parser.add_argument('--dim_hidden', default=1024, type=int)
#     parser.add_argument('--alpha', default = 1e-4, type = float)
    
#     args = parser.parse_args()

#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
#     time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
#     fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.log'.format(time_)))
#     fileHandler.setFormatter(logFormatter)
#     rootLogger.addHandler(fileHandler)
#     # args display
#     for k, v in vars(args).items():
#         logging.info(k+':'+str(v))

#     seed_everything(666)

#     train(args)


# if __name__ == '__main__':
#     main()
