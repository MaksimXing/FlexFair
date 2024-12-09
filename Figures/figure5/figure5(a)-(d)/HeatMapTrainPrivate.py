from FedBasicFunc.FedBasic import *
from util.Finetune import Finetune
from util.OOD.runOODmethod import runOODmethod
from util.Fairness.TestResultGeneration.ModelTestPrivate import RunTestMixedPrivate
from util.Fairness.TestResultGeneration.ModelTestPublic import RunTestMixedPublic
from util.Fairness.CalculateDPandEO.CalDPandEOPublic import calculateDP_PUBLIC
from util.Fairness.CalculateDPandEO.CalDPandEOPrivate import calculateDP_PRIVATE
from util.Draw.ParetoFront import *
from util.Draw.DrawHeatMapPrivate import runDraw

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    home_root = # data path here
    save_root = # output path here

    parser.add_argument('--datapath', type=str, default=home_root)
    parser.add_argument('--test_data_path', type=str, default=home_root)
    parser.add_argument('--savepath', type=str, default=save_root)

    parser.add_argument('--use_private_dataset', type=int, default=0)
    parser.add_argument('--zhongda', type=bool, default=True)
    parser.add_argument('--shengfuyou', type=bool, default=True)
    parser.add_argument('--shizhongliu', type=bool, default=True)
    parser.add_argument('--zhongzhong', type=bool, default=True)
    parser.add_argument('--use_public_dataset', type=int, default=1)
    parser.add_argument('--cvc', type=bool, default=True)
    parser.add_argument('--kva', type=bool, default=True)

    parser.add_argument('--mode', default='train')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--snapshot', default=None)
    parser.add_argument('--site_num', type=int, default=0,
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

    parser.add_argument('--dev_mode', type=bool, default=False, help='Choose develop mode')
    parser.add_argument('--two_stage_mode', type=bool, default=False, help='Choose 2 stage model')
    parser.add_argument('--mixup_mode', type=bool, default=False, help='Use Fair Mixup model')

    parser.add_argument('--miu', type=int, default=0.001,
                        help='Miu for FedProx')

    parser.add_argument('--rho', type=int, default=0,
                        help='Parameter controlling the momentum SGD, as well as FedNova')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Select which device to run')

    parser.add_argument('--test_dataset', type=str, default='CVC-300',
                        help='CVC-300, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, Kvasir')

    parser.add_argument('--save_model_per_epoch', type=int, default=1,
                        help='How many epochs to save model once')

    parser.add_argument('--start_save_model', type=int, default=0,
                        help='Which epoch to start saving model')

    parser.add_argument('--backbone', type=str, default='res2net50',
                        help='res2net50 / pvt_v2_b2')

    parser.add_argument('--start_test_epoch', type=int, default=75,
                        help='Which epoch to start test.py')

    parser.add_argument('--csv_path', type=str, default='',
                        help='where to save csv')

    parser.add_argument('--fairness_baseline', default='mixup', help='rex / mixup / gap')
    parser.add_argument('--mixup_alpha', default=1)
    parser.add_argument('--manifold_lam', default=1e-3)
    parser.add_argument('--manifold_clip', default=1)

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
    parser.add_argument('--use_private_color_exchange', default=1)



    args = parser.parse_args()

    args.rho = args.momentum

    arr = []
    if args.use_private_dataset:
        arr = [args.zhongda, args.shengfuyou, args.shizhongliu, args.zhongzhong]
    elif args.use_public_dataset:
        arr = [args.kva, args.cvc]
    args.site_num = np.sum(arr)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.target_GPU
    args.ft_lr = float(args.ft_lr)
    args.ft_epoch = int(args.ft_epoch)
    args.penalty_weight = float(args.penalty_weight)

    # model path here
    model_path_1 =
    model_path_2 =

    # test path here
    base_path =
    pic_path = ''
    gt_path = ''
    output_base_path =

    os.makedirs(output_base_path, exist_ok=True)

    for name in os.listdir(os.path.join(args.test_data_path, base_path, pic_path)):
        if 'label' in name:
            continue
        file_path = os.path.join(pic_path, name)

        label_name = os.path.splitext(name)[0] + '_label' + os.path.splitext(name)[1]

        output_pic_path = os.path.join(output_base_path, os.path.splitext(name)[0])

        runDraw(args, base_path, name, label_name, model_path_1, model_path_2, output_pic_path)
        print(f"Processed {name} and saved output to {output_pic_path}")