# F-SkinCancerClassification dataset

## Available Models and Methods

- FedAvg
- FedNova
- FedProx
- Scaffold
- FairMixup
- FairFed
- FlexFair(FedAvgOOD)


## preparation

* data：download `SkinDisease.zip`, and save them under the datapath
* model：download `SkinPretrain.zip`，and save it under the "models" folder
* fill in your data path in scripts/run.py

## Training Scripts

This project includes several training scripts for different models and fairness approaches. To run a single training script for a specific model, use one of the following commands:

bash scripts/run.sh

