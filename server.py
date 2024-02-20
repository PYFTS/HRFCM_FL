#import flwr as fl


# Start Flower server
#fl.server.start_server(
#    server_address="0.0.0.0:8080",
#    config=fl.server.ServerConfig(num_rounds=3),
#)

from typing import Dict, List, Tuple
import numpy as np

import flwr as fl
from flwr.common import Metrics


NUM_CLIENTS = 3
numRounds = 15

#def get_evaluate_fn(testset: testSet):
#    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
#    def evaluate(
#        server_round: int,
#        parameters: fl.common.NDArrays,
#        config: Dict[str, fl.common.Scalar],
#    ):
#        convLayersList = []
#        convLayersList.append([16, 3, 'same', 'relu', 1, True, 'max', None, False])
#        convLayersList.append([32, 3, 'same', 'relu', 1, True, 'max', None, False])
#        convLayersList.append([64, 3, 'same', 'relu', 1, True, 'max', None, False])
#
#        convLayers, denseLayers = createCNNLayers(convLayersList, getDataset.getTrainLoaders()[2])
#        learningRateList = [1e-8, 0.0001, 0.001, 0.01, 0.1]
#
#        initialModel = Sequential(layers.Rescaling(scale = 1./255, input_shape=getDataset.getTrainLoaders()[1]))
#        model = makeModel(initialModel, convLayers, denseLayers, learningRateList[2])
#        model.compile(optimizer='adam', #Adam(learning_rate=learningRate),
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                  metrics=['accuracy'])
#        #model = get_model()  # Construct the model
#        model.set_weights(parameters)  # Update model with the latest parameters
#        loss, accuracy = model.evaluate(testset, verbose=VERBOSE)
#        return loss, {"accuracy": accuracy}
#
#    return evaluate

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"rmse": sum(rmse) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    #evaluate_fn=get_evaluate_fn(testSet),  # global evaluation function
    )

# Start Flower server
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=numRounds),
    strategy=fl.server.strategy.FedAvg(),
)

#print(getDataset.datasetDir)
#print(history)

print(f"{history.metrics_centralized = }")

plotHistory.plot(history)