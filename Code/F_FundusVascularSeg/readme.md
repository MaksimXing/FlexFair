# F-FundusVascularSeg dataset

## Available Models and Methods

- FedAvg
- FedNova
- FedProx
- Scaffold
- FairMixup
- FairFed
- FlexFair(FedAvgOOD)



## preparation

* data：download `fundus_data.zip`, and save it under the "dataset" folder
* model：download `FundusPretrain.zip`，and save it under the "pretrain" folder


## Training Scripts

This project includes several training scripts for different methods. You can modify the script "source/run.sh" to run different models with different parameters, use the following command:

bash run.sh
