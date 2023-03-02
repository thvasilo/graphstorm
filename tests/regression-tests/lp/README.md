# Link Prediction regression tests

*Note*: the regression test is designed to run on GraphStorm docker environment. To learn how to configure your Linux environment to run the GraphStorm in docker, please refer to the [GraphStorm Onboarding Tutorial (Coming soon.)]. All below commands run within the GraphStorm docker container.

Prerequists
-----------
1. Make sure that the graph-storm codes are located in the /graph-storm folder within the docker environment.

2. set the GraphStorm Python path
```shell
export PYTHONPATH=/graph-storm/python
```

3. Configure the docker environment to allow it to access the target S3 bucket so that regression results could be saved to S3. For example, you can copy and paste the AK/SK bash command and run them in the container command line.


## MAG Link Prediction Regression Test
Original MAG graph is designed for node classification. In order to do link prediction, we set the "author,writes,paper" edge as the
target edge. Also for link prediction, we split these edges according to a percentage (0.8) for training and validation (8:2).


How to run:
-----------
Step 1: cd to the graph-storm folder

Step 2: create test data
```shell
sh -u tests/regression-tests/lp/prepare_ogbn_mag_lp_regression_test_data.sh
```

step 3: run the test
```shell
sh -u tests/regression-tests/lp/ogbn_mag_lp_regression_test.sh
```

Regression performance:
-----------------------

#### GSF v0.1
Train mrr: 0.9597, Val mrr: 0.3394, Test mrr: 0.4443,
Best val mrr: 0.3394, Best test mrr: 0.4443, Best iter: 3130


## Products Link Prediction Regression Test
OGBN Products is a homogeneous graph. For link prediction, we split its edges according to a percentage (0.8) for training and validation (8:2).


How to run:
-----------
Step 1: cd to the graph-storm folder

Step 2: create test data
```shell
sh -u tests/regression-tests/lp/prepare_ogbn_products_lp_regression_test_data.sh
```

step 3: run the test
```shell
sh -u tests/regression-tests/lp/ogbn_products_lp_regression_test.sh
```

Regression performance:
-----------------------
There is an issue about the Link Prediction task on the OGBN-Products data on the Open Source branch, i.e., if set to evaluate and save models in the same iteration, optimizer.step() will fail in the next iteration. So the below LP results on the GSF Open Source is based on the best results from three one-epoch runs.

#### GSF V0.1 (1 GPU)
Train mrr: -1.0000, Val mrr: 0.0912, Test mrr: 0.0900, Eval time: 1178.287, Best Evaluation step: 8800
