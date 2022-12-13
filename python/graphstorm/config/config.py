"""Builtin configs"""

BUILTIN_GNN_ENCODER = ["rgat", "rgcn"]
BUILTIN_ENCODER = ["lm"] + ["rgat", "rgcn"]
SUPPORTED_BACKEND = ["gloo", "nccl"]

BUILTIN_LP_LOSS_CROSS_ENTROPY = "cross_entropy"
BUILTIN_LP_LOSS_LOGSIGMOID_RANKING = "logsigmoid"
BUILTIN_LP_LOSS_FUNCTION = [BUILTIN_LP_LOSS_CROSS_ENTROPY, \
    BUILTIN_LP_LOSS_LOGSIGMOID_RANKING]

BUILTIN_TASK_NODE_CLASSIFICATION = "node_classification"
BUILTIN_TASK_NODE_REGRESSION = "node_regression"
BUILTIN_TASK_EDGE_CLASSIFICATION = "edge_classification"
BUILTIN_TASK_EDGE_REGRESSION = "edge_regression"
BUILTIN_TASK_LINK_PREDICTION = "link_prediction"
BUILTIN_TASK_MLM = "mlm"

SUPPORTED_TASKS  = [BUILTIN_TASK_NODE_CLASSIFICATION, \
    BUILTIN_TASK_NODE_REGRESSION, \
    BUILTIN_TASK_EDGE_CLASSIFICATION, \
    BUILTIN_TASK_LINK_PREDICTION, \
    BUILTIN_TASK_EDGE_REGRESSION, \
    BUILTIN_TASK_MLM]

EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY = "consecutive_increase"
EARLY_STOP_AVERAGE_INCREASE_STRATEGY = "average_increase"

# Task tracker
GRAPHSTORM_SAGEMAKER_TASK_TRACKER = "sagemaker_task_tracker"

SUPPORTED_TASK_TRACKER = [GRAPHSTORM_SAGEMAKER_TASK_TRACKER]