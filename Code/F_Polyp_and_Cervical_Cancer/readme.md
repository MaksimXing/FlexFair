# F-Polyp-and-Cervical-Cancer Dataset

## Available Models and Methods

- FedAvg
- FedNova
- FedProx
- Scaffold
- FairMixup
- FairFed
- FlexFair(FedAvgOOD)



## preparation

* data：download `train.zip` and `test.zip` (For Cervical Cancer) and `Public_5.zip` (For Polyp), and save them under the datapath
* model：download `Polyp_and_Cervical_pretrain.zip`，and save it under the "res" folder


## Training Scripts

This project includes several training scripts for different models and fairness approaches. To run a single training script for a specific model, use one of the following commands:


bash src/sh/fedavg-cervical.sh        ## Train the fedavg model in the cervical dataset
bash src/sh/fedavg-polyp.sh           ## Train the fedavg model in the polyp dataset

bash src/sh/fednova-cervical.sh       ## Train the fednova model in the cervical dataset
bash src/sh/fednova-polyp.sh          ## Train the fednova model in the polyp dataset

bash src/sh/fedprox-cervical.sh       ## Train the fedprox model in the cervical dataset
bash src/sh/fedprox-polyp.sh          ## Train the fedprox model in the polyp dataset

bash src/sh/scaffold-cervical.sh      ## Train the scaffold model in the cervical dataset
bash src/sh/scaffold-polyp.sh         ## Train the scaffold model in the polyp dataset

bash src/sh/mixup-cervical.sh         ## Train the fairmixup model in the cervical dataset
bash src/sh/mixup-polyp.sh            ## Train the fairmixup model in the polyp dataset

bash src/sh/fairfed-cervical.sh         ## Train the fairfed model in the cervical dataset
bash src/sh/fairfed-polyp.sh            ## Train the fairfed model in the polyp dataset

bash src/sh/flexfair-cervical.sh      ## Train our method in the cervical dataset
bash src/sh/flexfair-polyp.sh         ## Train our method in the polyp dataset


## note
For convenience, we named the cervical dataset private and the polyp dataset public.