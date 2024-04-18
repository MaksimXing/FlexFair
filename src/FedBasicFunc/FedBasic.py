# Fed import
from FedMethods.FedMethods import *
from FedBasicFunc.Data.FedData import *
from FedBasicFunc.Models.FedAvgModelForMixup import FedAvgModelForMixup

class Train(object):
    def __init__(self, args):

        # data
        DataList = []
        DataList.append(DataCVC)
        DataList.append(DataKVA)

        # Fairness mode
        if args.fairness_mode:
            self.alg = FedAvgOOD_Weighted_Public_REx(DataList, FedAvgModel, args)
            return
        # alg
        elif args.alg == 'fedavg':

            self.alg = FedAvgTrainPublic(DataList, FedAvgModel, args)
            return
        elif args.alg == 'fedprox':
            self.alg = FedProxTrainPublic(DataList, FedAvgModel, args)
            return
        elif args.alg == 'scaffold':
            self.alg = ScaffoldTrainPublic(DataList, FedAvgModel, args)
            return
        elif args.alg == 'fednova':
            self.alg = FedNovaTrainPublic(DataList, FedAvgModel, args)
            return
        else:
            self.alg = None

    def train(self):
        self.alg.train()


