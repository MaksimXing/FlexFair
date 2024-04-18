# FlexFair
FlexFair: Achieving Flexible Fairness Metrics in Federated Learning for Medical Image Analysis

**Remember to change path in `FedTrain.py`**

```python
home_root = 'your dataset path here'
save_root = 'your output path here'
```

### Run Baseline
```python
python FedTrain.py --epoch [epoch amount] --com_round [FL round] --alg [baseline method] --batch_size [batch size]
```
For example:
```python
python FedTrain.py --epoch 50 --com_round 5 --alg fedavg --batch_size 64
```

### Run FlexFair
```python
python FedTrain.py --epoch [epoch amount] --com_round [FL round] --fairness_mode 1 --fairness_step [number to split batch] --penalty_weight [penalty weight]
```
For example:
```python
python FedTrain.py --epoch 50 --com_round 5 --fairness_mode 1 --fairness_step 8 --penalty_weight 1.0
```
