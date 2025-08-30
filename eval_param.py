import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import os
import torch
import pprint
import json
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models.utils.helper_utils import optimizer_to
import debugpy


torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    if os.getenv('debug'):
        debugpy.listen(5678)
        debugpy.wait_for_client()
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='data/splits/oct21.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true', default=True)
    parser.add_argument('--model_arch', help='model to use', default='seq2seq_im_mask')
    parser.add_argument('--gpu', help='use gpu', action='store_true', default=True)
    parser.add_argument('--dout', help='where to save model')
    parser.add_argument('--resume', help='load a checkpoint')

    # hyper parameters
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=5*7*7, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0, type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0.2, type=float)

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)

    # other settings
    parser.add_argument('--dec_teacher_forcing', help='enable decoder teacher forcing', action='store_true', default=True)
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')
    parser.add_argument('--panoramic', help='use panoramic', action='store_true', default=True)
    parser.add_argument('--orientation', help='use orientation features', action='store_true')
    parser.add_argument('--panoramic_concat', help='use panoramic', action='store_true', default=True)

    # CL args
    parser.add_argument("--mode", type=str, help="Select CIL method")
    parser.add_argument("--n_tasks", type=int, help="The number of tasks")
    parser.add_argument("--rnd_seed", type=int, default=1, help="Random seed number.")
    parser.add_argument("--stream_seed", type=int, help="Random seed number for stream.")
    parser.add_argument("--memory_size", type=int, default=500, help="Episodic memory size (# videos)")
    parser.add_argument("--incremental_setup", type=str, choices=['behavior_il','behavior_il_test', 'environment_il', 'environment_il_nosampling'])

    # Dataset
    parser.add_argument("--log_path", type=str, default="results", help="The path logs are saved.")

    # Model
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")

    # Train
    parser.add_argument("--opt_name", type=str, default="adam", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--batchsize", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs_per_task", type=int, default=10, help="number of epochs for each task (offline per-task training)")

    # Regularization
    parser.add_argument("--reg_coef", type=int, default=100, help="weighting for the regularization loss term")
    parser.add_argument("--data_dir", type=str, help="location of the dataset")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")

    # Eval period
    parser.add_argument("--eval_period", type=int, default=500000, help="evaluation period for true online setup")

    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.") # 1.182575531717415 0.8456119488179613

    # CLIB
    parser.add_argument("--imp_update_period", type=int, default=1, help="period between importance update, in units of model updates (increase for heavy datasets like ImageNet)")
    parser.add_argument('--lr_step', type=float, default=0.95, help='step of iterating lr for adaptive LR')
    parser.add_argument('--lr_length', type=int, default=10, help='period of iterating lr for adaptive LR')
    parser.add_argument('--lr_period', type=int, default=10, help='period of iterating lr for adaptive LR')

    # MIR
    parser.add_argument('--mir_cands', type=int, default=50, help='# candidates to use for MIR')

    # CAMA
    parser.add_argument('--N', type=int, default=5, help='# the recent confidence scores for CAMA')

    # XDER
    parser.add_argument("--simclr_batch_size", type=int, default=64)
    parser.add_argument("--simclr_num_aug", type=int, default=2)
    parser.add_argument("--simclr_temp", type=int, default=5)
    parser.add_argument("--total_classes", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--param_m", type=float, default=0.7)
    parser.add_argument("--lambd", type=float, default=0.05)

    # SD-LoRA
    parser.add_argument("--lora_rank", type=int, default=10, help="Rank for LoRA adaptation")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="Alpha parameter for LoRA")
    parser.add_argument("--adaptation_lr", type=float, default=1e-4, help="Learning rate for LoRA parameters")
    parser.add_argument("--ortho_reg_weight", type=float, default=0.1, help="Weight for orthogonal regularization")
    # O-LoRA regularization weights
    parser.add_argument("--lamda_1", type=float, default=0.5, help="Orthogonal regularization weight (O-LoRA)")
    parser.add_argument("--lamda_2", type=float, default=0.0, help="L2 regularization weight for loranew params (O-LoRA)")

    # args and init
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CL-ALFRED n_task
    if args.incremental_setup in ['behavior_il','behavior_il_test']:
        args.n_tasks = 7
    elif args.incremental_setup in ['environment_il']:
        args.n_tasks = 4
    else:
        raise Exception("args.n_task should be either 7 (Behavior-IL) or 4 (Environment-IL).")

    # check if dataset has been preprocessed
    if not os.path.exists(os.path.join(args.data, "%s.vocab" % args.pp_folder)) and not args.preprocess:
        raise Exception("Dataset not processed; run with --preprocess")

    # make output dir
    pprint.pprint(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})

    # preprocess and save
    if args.preprocess:
        print("\nPreprocessing dataset and saving to %s folders ... This will take a while. Do this once as required." % args.pp_folder)
        dataset = Dataset(args, None)
        dataset.preprocess_splits(splits)
        vocab = torch.load(os.path.join(args.dout, "%s.vocab" % args.pp_folder))
    else:
        vocab = torch.load(os.path.join(args.data, "%s.vocab" % args.pp_folder))

    # load model
    # M = import_module('model.{}'.format(args.model_arch))
    # if args.resume:
    #     print("Loading: " + args.resume)
    #     model, optimizer = M.Module.load(args.resume)
    # else:
    #     model = M.Module(args, vocab)
    #     optimizer = None

    # # to gpu
    # if args.gpu:
    #     model = model.to(torch.device('cuda'))
    #     if not optimizer is None:
    #         optimizer_to(optimizer, torch.device('cuda'))

    ############################
    # print('='*40)
    # print(f'model_name: {args.resume}')
    # 计算并打印可训练参数的总数
    # print('Total # trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    ############################
    
    # 打印param的名字和尺寸
    # for name, param in model.named_parameters():
    #     # if param.requires_grad:
    #     print(name, param.size(),f' - requires_grad: {param.requires_grad}')
    #     # 打印enc_att_goal.scorer.lora_B的value
    #     if  'enc_att_goal.scorer.lora_B' in name:
    #         print('lora_B value:', param)


    
    
    model1_path = 'exp/behavior_il_new/er/s1/net_epoch_000022510_look_at_obj_in_light.pth'
    M1 = import_module('models.model.seq2seq_im_mask')
    model2_path = 'exp/behavior_il_new/olora/s1/net_epoch_000022510_look_at_obj_in_light.pth'
    M2 = import_module('models.model.seq2seq_im_mask_olora')

    model1, optimizer1 = M1.Module.load(model1_path)
    model2, optimizer2 = M2.Module.load(model2_path)
    # print(optimizer1)
    # print('-'*40)
    # print(optimizer2)
    # print('-'*40)

    # 对比enc_att_goal.scorer.weight 和enc_att_goal.scorer.lora_A @ enc_att_goal.scorer.lora_B+enc_att_goal.scorer.original_layer.weight 的差异
    # 从model1中提取enc_att_goal.scorer.weight
    for name1, param1 in model1.named_parameters():
        if  'enc_att_goal.scorer.weight' in name1:
            weight1 = param1
            print('From model1:', name1, weight1.size())
            break
    
    # 从model2中提取enc_att_goal.scorer.lora_A, enc_att_goal.scorer.lora_B, enc_att_goal.scorer.original_layer.weight
    for name2, param2 in model2.named_parameters():
        if  'enc_att_goal.scorer.lora_A' in name2:
            lora_A = param2
            print('From model2:', name2, lora_A.size())
        elif 'enc_att_goal.scorer.lora_B' in name2:
            lora_B = param2
            print('From model2:', name2, lora_B.size())
        elif 'enc_att_goal.scorer.original_layer.weight' in name2:
            original_weight = param2
            print('From model2:', name2, original_weight.size())
    
    # 计算lora_A @ lora_B
    lora_product = torch.matmul(lora_B, lora_A)
    print('lora_A @ lora_B size:', lora_product.size())
    # 计算lora_A @ lora_B + original_weight
    combined_weight = lora_product + original_weight
    print('Combined weight size:', combined_weight.size())

    # 打印前20个元素进行对比
    print('First 20 elements of model1 weight:', weight1.view(-1)[:20])
    print('First 20 elements of combined weight:', combined_weight.view(-1)[:20])

    # 计算weight1和combined_weight的差异
    if weight1.size() == combined_weight.size():
        difference = torch.norm(weight1 - combined_weight).item()
        print('Difference (Frobenius norm) between model1 weight and combined weight:', difference)

