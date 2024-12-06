from method import *
from DrawSurface import DrawLossSurface
from DrawHeatMap import runDraw

output_path = '../output/'

if __name__ == '__main__':
    class Args:
        def __init__(self):
            ## set the backbone type
            self.backbone = 'res2net50'
            ## set the path of training dataset
            self.datapaths = ['../dataset/CHASEDB1', '../dataset/DRIVE', '../dataset/STARE']
            self.datapath = ''
            self.dataset = ['CHASEDB1', 'DRIVE', 'STARE']
            ## set the path of logging
            self.output_path = output_path
            ## keep unchanged
            self.mode = 'train'
            self.epoch = 46
            self.batch_size = 4
            self.lr = 0.4
            self.num_workers = 4
            self.weight_decay = 1e-3
            self.clip = 0.5
            self.com_round = 3
            self.miu = 0.01
            self.rho = 0.9
            self.fairness_step = 5
            self.penalty_weight = 0.5

            self.use_swa = True
            self.swa_start = 0.75
            self.rho_sam = 0.05
            self.eta_sam = 0.01


    ## training
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    args = Args()



    model_path_1 = # put model path here
    model_path_2 =

    base_path = # test set path here
    pic_path =
    gt_path =
    output_base_path =
    gt_attach = # if pic and ground truth name are different

    os.makedirs(output_base_path, exist_ok=True)

    for name in os.listdir(os.path.join(base_path, pic_path)):
        print(name)
        file_path = os.path.join(base_path, pic_path, name)

        output_pic_path = os.path.join(output_base_path, os.path.splitext(name)[0])

        runDraw(args, base_path, pic_path, gt_path, name, model_path_1, model_path_2, output_pic_path, gt_attach)
        print(f"Processed {name} and saved output to {output_pic_path}")
