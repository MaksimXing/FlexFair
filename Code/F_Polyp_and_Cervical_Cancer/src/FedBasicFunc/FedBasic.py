# Fed import
from FedMethods.FedMethods import *
from FedBasicFunc.Data.FedData import *
# from FedBasicFunc.Models.FedAvgModelForMixup import FedAvgModelForMixup

class Train(object):
    def __init__(self, args):

        # data
        DataList = []
        if args.use_private_dataset:
            if args.shengfuyou:
                DataList.append(DataShengfuyou)
            if args.newzhongda:
                DataList.append(DataNewZhongda)
            if args.newzhongyi:
                DataList.append(DataNewZhongyi)
            if args.newzhongzhong:
                DataList.append(DataNewZhongzhong)
        elif args.use_public_dataset:
            if args.cvc:
                DataList.append(DataCVC)
            if args.kva:
                DataList.append(DataKVA)

        # Fairness mode
        if args.fairness_mode:
            if args.use_weight:
                    if args.use_public_dataset:
                        if args.fairness_baseline == 'rex':
                            self.alg = FedAvgOOD_Weighted_Public_REx(DataList, FedAvgModel, args)
                            return
                    elif args.use_private_dataset:
                        if args.fairness_baseline == 'rex':
                            self.alg = FedAvgOOD_Weighted_Private_REx(DataList, FedAvgModel, args)
                            return

        # alg
        elif args.alg == 'fedavg':
            if args.use_public_dataset:
                self.alg = FedAvgTrainPublic(DataList, FedAvgModel, args)
                return
            elif args.use_private_dataset:
                self.alg = FedAvgTrainPrivate(DataList, FedAvgModel, args)
                return
        elif args.alg == 'fedprox':
            if args.use_public_dataset:
                self.alg = FedProxTrainPublic(DataList, FedProxModel, args)
                return
            elif args.use_private_dataset:
                self.alg = FedProxTrainPrivate(DataList, FedProxModel, args)
                return
        elif args.alg == 'scaffold':
            if args.use_public_dataset:
                self.alg = ScaffoldTrainPublic(DataList, ScaffoldModel, args)
                return
            elif args.use_private_dataset:
                self.alg = ScaffoldTrainPrivate(DataList, ScaffoldModel, args)
                return
        elif args.alg == 'fednova':
            if args.use_public_dataset:
                self.alg = FedNovaTrainPublic(DataList, FedNovaModel, args)
                return
            elif args.use_private_dataset:
                self.alg = FedNovaTrainPrivate(DataList, FedNovaModel, args)
                return
        elif args.alg == 'fairmixup':
            if args.use_public_dataset:
                self.alg = FairMixupTrainPublic(DataList, FairMixupModel, args)
                return
            elif args.use_private_dataset:
                self.alg = FairMixupTrainPrivate(DataList, FairMixupModel, args)
                return
        elif args.alg == 'fairfed':
            if args.use_public_dataset:
                self.alg = FairFedTrainPublic(DataList, FairMixupModel, args)
                return
            elif args.use_private_dataset:
                self.alg = FairFedTrainPrivate(DataList, FairMixupModel, args)
                return            
        else:
            self.alg = None

    def train(self):
        self.alg.train()


