import argparse
import json
import os
import copy
import time
from datetime import datetime
import logging
from tqdm import *

from torch.utils.tensorboard import SummaryWriter

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv


def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)

    data_prefix = './data/' + func_args.market + '/'
    relation_file = func_args.relation_file if func_args.relation_file else 'industry_classification.npy'
    matrix_path = data_prefix + relation_file

    start_time = datetime.now().strftime('%m%d_%H%M%S')  # 콜론을 밑줄로 대체
    if func_args.mode == 'train':
        PREFIX = 'outputs/'
        PREFIX = os.path.join(PREFIX, start_time)
        img_dir = os.path.join(PREFIX, 'img_file')
        save_dir = os.path.join(PREFIX, 'log_file')
        model_save_dir = os.path.join(PREFIX, 'model_file')

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)

        hyper = copy.deepcopy(func_args.__dict__)
        print(hyper)
        hyper['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        json_str = json.dumps(hyper, indent=4)

        with open(os.path.join(save_dir, 'hyper.json'), 'w') as json_file:
            json_file.write(json_str)

        writer = SummaryWriter(save_dir)
        writer.add_text('hyper_setting', str(hyper))

        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('WARNING')
        fhlr = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)

        logger.info("Starting training...")

        if func_args.market == 'DJIA':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            logger.info(f"Loaded Stocks Data Shape: {stocks_data.shape}")
            logger.info(f"Loaded Rate of Return Data Shape: {rate_of_return.shape}")
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(hyper['device'])
            test_idx = 7328
            allow_short = True
        elif func_args.market == 'HSI':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            logger.info(f"Loaded Stocks Data Shape: {stocks_data.shape}")
            logger.info(f"Loaded Rate of Return Data Shape: {rate_of_return.shape}")
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(hyper['device'])
            test_idx = 4211
            allow_short = True

        elif func_args.market == 'CSI100':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            logger.info(f"Loaded Stocks Data Shape: {stocks_data.shape}")
            logger.info(f"Loaded Rate of Return Data Shape: {rate_of_return.shape}")
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(hyper['device'])
            test_idx = 1944
            market_history = None
            allow_short = False

        func_args.num_assets = stocks_data.shape[0]

        env = PortfolioEnv(assets_data=stocks_data, market_data=market_history, rtns_data=rate_of_return,
                           in_features=func_args.in_features, val_idx=test_idx, test_idx=test_idx,
                           batch_size=func_args.batch_size, window_len=func_args.window_len, trade_len=func_args.trade_len,
                           max_steps=func_args.max_steps, mode=func_args.mode, norm_type=func_args.norm_type,
                           allow_short=allow_short)

        supports = [A]
        actor = RLActor(supports, func_args).to(hyper['device'])
        agent = RLAgent(env, actor, func_args)

        mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))
        try:
            max_cr = 0
            for epoch in range(func_args.epochs):
                epoch_return = 0
                for j in tqdm(range(mini_batch_num)):
                    # 여기에 필요한 코드를 추가하세요
                    pass
                logger.info(f"Epoch {epoch+1}/{func_args.epochs} completed.")
                writer.add_scalar('Loss/train', epoch_return, epoch)  # 스칼라 데이터 기록
        except KeyboardInterrupt:
            logger.warning("Training interrupted")

        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--window_len', type=int, default=20)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false')
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false')
    parser.add_argument('--relation_file', type=str, default='industry_classification.npy')
    parser.add_argument('--addaptiveadj', dest='addaptiveadj', action='store_true')
    parser.add_argument('--market', type=str, default='DJIA')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--in_features', type=int, nargs='+', default=[6, 4])
    parser.add_argument('--trade_len', type=int, default=21)
    parser.add_argument('--max_steps', type=int, default=12)
    parser.add_argument('--norm_type', type=str, default='div-last')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1500)
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run(args)