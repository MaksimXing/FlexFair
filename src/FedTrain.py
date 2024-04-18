from FedBasicFunc.FedBasic import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    home_root = 'your dataset path here'
    save_root = 'your output path here'


    parser.add_argument('--datapath', type=str, default=home_root)
    parser.add_argument('--test_data_path', type=str, default=home_root)
    parser.add_argument('--savepath', type=str, default=save_root)

    parser.add_argument('--use_public_dataset', type=int, default=1)
    parser.add_argument('--cvc', type=bool, default=True)
    parser.add_argument('--kva', type=bool, default=True)

    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--snapshot', default=None)
    parser.add_argument('--site_num', type=int, default=2,
                        help='how many sites (different datasets) are involved)')
    parser.add_argument('--is_same_initial', type=bool, default=True,
                        help='Whether initial all the models with the same parameters in fedavg')

    parser.add_argument('--lr', default=0.4)
    parser.add_argument('--epoch', default=50)
    parser.add_argument('--com_round', type=int, default=5, help='number of maximum communication round')
    parser.add_argument('--use_color_exchange', type=int, default=1, help='Use CE module in SANet')

    parser.add_argument('--alg', type=str, default='fedavg',
                        help='fl algorithms: fedavg/fedprox/scaffold/fednova')
    parser.add_argument('--target_GPU', type=str, default='0', help='Choose which GPU to use')


    parser.add_argument('--miu', type=int, default=0.001,
                        help='Miu for FedProx')
    parser.add_argument('--rho', type=int, default=0,
                        help='Parameter controlling the momentum SGD, as well as FedNova')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Select which device to run')
    parser.add_argument('--save_model_per_epoch', type=int, default=1,
                        help='How many epochs to save model once')
    parser.add_argument('--start_save_model', type=int, default=0,
                        help='Which epoch to start saving model')
    parser.add_argument('--backbone', type=str, default='res2net50',
                        help='res2net50')
    parser.add_argument('--start_test_epoch', type=int, default=0,
                        help='Which epoch to start test.py')

    parser.add_argument('--penalty_weight', default=1.0)
    parser.add_argument('--fairness_mode', default=0, help='Choose Fairness model')
    parser.add_argument('--fairness_step', default=0)

    args = parser.parse_args()

    # FedNova
    args.rho = args.momentum


    os.environ['CUDA_VISIBLE_DEVICES'] = args.target_GPU
    args.penalty_weight = float(args.penalty_weight)
    args.epoch = int(args.epoch)
    args.batch_size = int(args.batch_size)
    args.com_round = int(args.com_round)
    args.fairness_mode = int(args.fairness_mode)
    args.fairness_step = int(args.fairness_step)

    t = Train(args)
    t.train()
