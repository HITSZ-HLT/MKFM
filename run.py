import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import IEMOCAPDataset
from model import *
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from trainer import  train_or_eval_model, save_badcase
from dataset import IEMOCAPDataset
from dataloader import get_IEMOCAP_loaders
from transformers import AdamW
import copy
import tqdm
from losses import st_SCL

seed = 100

import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    path = './saved_models/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model_dir', type=str, default='')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='')


    parser.add_argument('--bert_dim', type = int, default=1024)
    parser.add_argument('--hidden_dim', type = int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')

    parser.add_argument('--attn_type', type=str, default='rgcn', choices=['dotprod','linear','bilinear', 'rgcn'], help='Feature size.')
    parser.add_argument('--no_rel_attn',  action='store_true', default=False, help='no relation for edges' )

    parser.add_argument('--max_sent_len', type=int, default=200,
                        help='max content length for each text, if set to 0, then the max length has no constrain')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='IEMOCAP', type= str, help='dataset name, IEMOCAP or MELD or DailyDialog')

    parser.add_argument('--windowp', type=int, default=1,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowp_aadj', type=int, default=1,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=0,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--nodal_att_type', type=str, default=None, choices=['global','past'], help='type of nodal attention')

    
    # 标签增强知识
    parser.add_argument('--TP', action='store_true', default=False, help='does not use topic')
    parser.add_argument('--SC', action='store_true', default=False, help='does not use scarcasm')
    parser.add_argument('--MP', action='store_true', default=False, help='does not use metaphor')

    # 标签语义知识
    parser.add_argument('--EC', action='store_true', default=False, help='does not use emtional cause')
    parser.add_argument('--CS', action='store_true', default=False, help='does not use common')
    parser.add_argument('--ACS', action='store_true', default=False, help='does not use affective common')

    # 标签上下文知识
    parser.add_argument('--CR', action='store_true', default=False, help='does not use co-reference')
    parser.add_argument('--CT', action='store_true', default=False, help='does not use context')
    parser.add_argument('--EC2', action='store_true', default=False, help='does not use emtional cause2')

    parser.add_argument('--seed', type=int, default=-1, help='seeds')

    args = parser.parse_args()
    print(args)
    
    seed_everything()
    
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    str1 = 'TP' if args.TP else "0"
    str2 = 'SC' if args.SC else "0"
    str3 = 'MP' if args.MP else "0"
    str4 = 'EC' if args.EC else "0"
    str5 = 'CS' if args.CS else "0"
    str6 = 'ACS' if args.ACS else "0"
    str7 = 'CR' if args.CR else "0"
    str8 = 'CT' if args.CT else "0"
    str9 = 'EC2' if args.EC2 else "0"

    logger = get_logger(path + args.dataset_name + '/logging.log')
    logger.info('start training on GPU {}!'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info(args)

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    
    ss_l = [0]#, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if args.seed == -1:
        ss_l2 = [0, 1, 2, 3, 4]
    else:
        ss_l2 = [args.seed]
    # ss = 0

    for xs in ss_l:
        for ss2 in ss_l2:
            start_time = time.time()
            seed_everything(100 + 100 + ss2)
            print(100 + 100 + ss2)
            # args.windowp = ss2

            train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec = get_IEMOCAP_loaders(dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args = args)
            n_classes = len(label_vocab['itos'])
        
            print('building model..')
            model = DAGERC_fushion(args, n_classes)


            if torch.cuda.device_count() > 1:
                print('Multi-GPU...........')
                model = nn.DataParallel(model,device_ids = range(torch.cuda.device_count()))
            if cuda:
                model.cuda()

            loss_function = nn.CrossEntropyLoss(ignore_index=-1)
            stSCL = st_SCL()
            optimizer = AdamW(model.parameters() , lr=args.lr)

            best_fscore,best_acc, best_loss, best_label, best_pred, best_mask = None,None, None, None, None, None
            all_fscore, all_acc, all_loss = [], [], []
            best_acc = 0.
            best_fscore = 0.
            best_valid_fscore = 0.
            best_model = None
            for e in range(n_epochs):

                if args.dataset_name=='DailyDialog':
                    train_loss, train_acc, _, _, train_micro_fscore, train_macro_fscore = train_or_eval_model(model, loss_function, stSCL,
                                                                                                        train_loader, e, cuda,
                                                                                                        args, optimizer, True)
                    valid_loss, valid_acc, _, _, valid_micro_fscore, valid_macro_fscore = train_or_eval_model(model, loss_function, stSCL,
                                                                                                        valid_loader, e, cuda, args)
                    test_loss, test_acc, test_label, test_pred, test_micro_fscore, test_macro_fscore = train_or_eval_model(model,loss_function, stSCL, test_loader, e, cuda, args)

                    all_fscore.append([valid_micro_fscore, test_micro_fscore, valid_macro_fscore, test_macro_fscore])

                    logger.info( 'Epoch: {}, train_loss: {}, train_acc: {}, train_micro_fscore: {}, train_macro_fscore: {}, valid_loss: {}, valid_acc: {}, valid_micro_fscore: {}, valid_macro_fscore: {}, test_loss: {}, test_acc: {}, test_micro_fscore: {}, test_macro_fscore: {}, time: {} sec'. \
                            format(e + 1, train_loss, train_acc, train_micro_fscore, train_macro_fscore, valid_loss, valid_acc, valid_micro_fscore, valid_macro_fscore, test_loss, test_acc,
                                test_micro_fscore, test_macro_fscore, round(time.time() - start_time, 2)))
                    
                else:

                    if args.dataset_name=='MELD':
                        ss = [0.3, 0.5, 0.8]

                    if args.dataset_name=='IEMOCAP':
                        ss = [0.5, 0.8, 1.0]

                    if args.dataset_name=='EmoryNLP':
                        ss = [0.2, 0.5, 0.8]
                    # ss = [xs, xs, xs]


                    train_loss, train_acc, _, _, train_fscore = train_or_eval_model(ss, model, loss_function, stSCL,
                                                                                    train_loader, e, cuda,
                                                                                    args, optimizer, True)
                    valid_loss, valid_acc, _, _, valid_fscore= train_or_eval_model(ss, model, loss_function, stSCL,
                                                                                    valid_loader, e, cuda, args)
                    test_loss, test_acc, test_label, test_pred, test_fscore= train_or_eval_model(ss, model,loss_function, stSCL, test_loader, e, cuda, args)

                    all_fscore.append([valid_fscore, test_fscore])
          
                    logger.info( 'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                         format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                         test_fscore, round(time.time() - start_time, 2)))

                #torch.save(model.state_dict(), path + args.dataset_name + '/model_' + str(e) + '_' + str(test_acc)+ '.pkl')

                if valid_fscore > best_valid_fscore:
                    best_valid_fscore = valid_fscore

                # if test_fscore > best_fscore:
                #     torch.save(model.state_dict(), path + args.dataset_name + '/model_' + str(e) + '_' + str( (100 + 100 + ss2) ) + '_' + str(test_fscore) + '_' + str(str1) + '_' + str(str2) + '_' + str(str3) + '_' + str(str4) + '_' + str(str5) + '_' + str(str6) + '_' + str(str7) + '_' + str(str8) + '_' + str(str9) + '.pkl')

                #     best_fscore = test_fscore

                e += 1


            if args.tensorboard:
                writer.close()

            logger.info('finish training!')

            #print('Test performance..')
            all_fscore = sorted(all_fscore, key=lambda x: (x[0],x[1]), reverse=True)
            #print('Best F-Score based on validation:', all_fscore[0][1])
            #print('Best F-Score based on test:', max([f[1] for f in all_fscore]))

            #logger.info('Test performance..')
            #logger.info('Best F-Score based on validation:{}'.format(all_fscore[0][1]))
            #logger.info('Best F-Score based on test:{}'.format(max([f[1] for f in all_fscore])))

            if args.dataset_name=='DailyDialog':
                logger.info('Best micro/macro F-Score based on validation:{}/{}'.format(all_fscore[0][1],all_fscore[0][3]))
                all_fscore = sorted(all_fscore, key=lambda x: x[1], reverse=True)
                logger.info('Best micro/macro F-Score based on test:{}/{}'.format(all_fscore[0][1],all_fscore[0][3]))
            else:
                logger.info('Best validation F-Score:{}'.format(best_valid_fscore))
                logger.info('Best F-Score based on validation:{}'.format(all_fscore[0][1]))
                logger.info('Best F-Score based on test:{}'.format(max([f[1] for f in all_fscore])))

            print(round(time.time() - start_time, 2))

            #save_badcase(best_model, test_loader, cuda, args, speaker_vocab, label_vocab)
