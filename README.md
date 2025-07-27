Taking the CIFAR-10 dataset as an example, you can use the following command to reproduce the experimental results.

```python
python main.py -dataset cifar10 -end_model cnn -edge_mode resnet10 -frac 1 -num_edges 4 -num_ends 80 -num_loc_tra 1 -num_comm 100 -alg FedConProAgg
```







