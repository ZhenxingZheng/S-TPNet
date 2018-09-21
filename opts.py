import argparse

parser = argparse.ArgumentParser(description='improved TRN')

#======================Code Configs=========================
parser.add_argument('--test_video_list', default='list/hmdb_split1_test.txt', type=str)
parser.add_argument('--root', default='./Datasets/', type=str)
parser.add_argument('--dataset', default='hmdb', type=str)
parser.add_argument('--log_dir', default='log', type=str)
parser.add_argument('--model_dir', default='model', type=str)
parser.add_argument('--get_scores', default=False, type=bool)
parser.add_argument('--description', type=str)


#===================Learning Configs====================
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=str)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--lr_step', default=40, type=int)
parser.add_argument('--print_freq', default=20, type=int)
parser.add_argument('--eval_freq', default=1, type=int)


#====================Model Configs======================
parser.add_argument('--segments', default=3, type=int)
parser.add_argument('--frames', default=4, type=int)
parser.add_argument('--base_model', default='resnet34', type=str)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--d_model', default=512, type=int)
parser.add_argument('--start', default=2, type=int)
