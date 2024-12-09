from FedBasicFunc.FedBasic import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # fill in your path
    home_root =
    save_root = '../public/output'

    parser.add_argument('--datapath', type=str, default=home_root)
    parser.add_argument('--test_data_path', type=str, default=home_root)
    parser.add_argument('--savepath', type=str, default=save_root)
    
    parser.add_argument('--use_private_dataset', type=int, default=0)
    parser.add_argument('--shengfuyou', type=bool, default=False)
    parser.add_argument('--newzhongda', type=bool, default=False)
    parser.add_argument('--newzhongzhong', type=bool, default=False)
    parser.add_argument('--newzhongyi', type=bool, default=False)

    parser.add_argument('--use_public_dataset', type=int, default=1)
    parser.add_argument('--cvc', type=bool, default=True)
    parser.add_argument('--kva', type=bool, default=True)
    
    parser.add_argument('--mode', default='train')
    
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nesterov', default=True)
    
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--snapshot', default=None)
    # site count involved in training, no need to change
    parser.add_argument('--site_num', type=int, default=0,
                        help='how many sites (different datasets) are involved)')
    parser.add_argument('--is_same_initial', type=bool, default=True,
                        help='Whether initial all the models with the same parameters in fedavg')

    
    parser.add_argument('--lr', default=0.4)
    parser.add_argument('--epoch', default=30)
    parser.add_argument('--com_round', type=int, default=1, help='number of maximum communication round')
    # Color exchange (SANet) for Polyp
    parser.add_argument('--use_color_exchange', type=int, default=1, help='Use CE module in SANet')
    
    # algorithm use
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='fl algorithms: fedavg/fedprox/scaffold/fednova')
    parser.add_argument('--target_GPU', type=str, default='0', help='Choose which GPU to use')

    
    # FredProx para 
    parser.add_argument('--miu', type=float, default=0.001, help='Miu for FedProx')
    # FedNova para 
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD, as well as FedNova')
    # Scaffold para 
    parser.add_argument('--c_lr', default=0.9)
    # FairFed para 
    parser.add_argument('--beta', type=float, default=0.9)


    # CPU/GPU
    parser.add_argument('--device', type=str, default='cuda:0', help='Select which device to run')
    parser.add_argument('--save_model_per_epoch', type=int, default=1, help='How many epochs to save model once')
    parser.add_argument('--start_save_model', type=int, default=100, help='Which epoch to start saving model')
    parser.add_argument('--backbone', type=str, default='res2net50', help='res2net50 / pvt_v2_b2')
    parser.add_argument('--start_test_epoch', type=int, default=100, help='Which epoch to start test.py')
    parser.add_argument('--csv_path', type=str, default='', help='where to save csv')
    
    parser.add_argument('--seed', type = int, default=0)
    
    

    # penalty weight on FlexFair
    parser.add_argument('--penalty_weight', default=1.0)
    parser.add_argument('--fairness_loss_mode', default='ce_loss', help='ce_loss / dice_loss / total_loss')
    parser.add_argument('--ft_epoch', default=1)
    parser.add_argument('--ft_lr', default=1e-5)
    parser.add_argument('--ft_is_std', default=1)

    parser.add_argument('--only_correspond_testset', default=1)
    parser.add_argument('--temp_input', default=None)
    parser.add_argument('--ood_use_fed', default=0)
    parser.add_argument('--fairness_mode', default=0, help='Choose Fairness model')
    parser.add_argument('--fairness_epoch', default=75)
    parser.add_argument('--fairness_target', default='site', help='site / rgb / Scale / Lightness')
    parser.add_argument('--use_weight', default=1)
    parser.add_argument('--fairness_step', default=0)
    parser.add_argument('--use_private_color_exchange', default=0)

    
    parser.add_argument('--method', default='fairmixup')
    

    # init args
    args = parser.parse_args()
    
    # Random Seed Fixed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    

    # auto detect site_num
    arr = []
    if args.use_private_dataset:
        arr = [args.shengfuyou, args.newzhongda, args.newzhongzhong, args.newzhongyi]
    elif args.use_public_dataset:
        arr = [args.kva, args.cvc]
    args.site_num = np.sum(arr)
    
    # auto detect and reset
    os.environ['CUDA_VISIBLE_DEVICES'] = args.target_GPU
    args.ft_lr = float(args.ft_lr)
    args.ft_epoch = int(args.ft_epoch)
    args.penalty_weight = float(args.penalty_weight)


    if args.method == 'fedavg':
        args.alg = 'fedavg'
        weight = 0
    elif args.method == 'fednova':
        args.alg = 'fednova'
        weight = args.rho
    elif args.method == 'fedprox':
        args.alg = 'fedprox'
        weight = args.miu
    elif args.method == 'scaffold':
        args.alg = 'scaffold'
        weight = args.c_lr
    elif args.method == 'fairmixup':
        args.alg = 'fairmixup'
        weight = args.penalty_weight
    elif args.method == 'fairfed':
        args.alg = 'fairfed'
        weight = args.beta
    elif args.method == 'flexfair':
        args.alg = 'fedavg'
        args.fairness_mode = 1
        args.fairness_baseline = 'rex'
        args.use_weight = 1
        args.fairness_step = 8
        args.batch_size = 32
        weight = args.penalty_weight
    
    args.savepath = f'../output/public/{args.method}/s{seed}_w{weight}/'
    t = Train(args)
    t.train()  
    