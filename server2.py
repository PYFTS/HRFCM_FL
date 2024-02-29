#import flwr as fl


# Start Flower server
#fl.server.start_server(
#    server_address="0.0.0.0:8080",
#    config=fl.server.ServerConfig(num_rounds=3),
#)
from logging import WARNING
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
import copy

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg #aggregate_inplace
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy, fedavg
from flwr.common import Metrics
from flwr.common.logger import log
import FCM_FTS
from pyFTS.benchmarks import Measures
from pyFTS.fcm import Activations
from pyFTS.partitioners import Grid
from pyFTS.common import Membership as mf

import lossFunction
from aggregate_inplace import aggregate_inplace

NUM_CLIENTS = 3
numRounds = 3

# Create clients partition
df1 = pd.read_csv('nrel_DHHL_1.csv')
df2 = pd.read_csv('nrel_DHHL_2.csv')
df3 = pd.read_csv('nrel_DHHL_3.csv')
#df4 = pd.read_csv('https://query.data.world/s/56i2vkijbvxhtv5gagn7ggk3zw3ksi', sep=';')

serverSet = {}
serverSet[0] = df1['value'].values[8000:10000]
serverSet[1] = df2['value'].values[8000:10000]
serverSet[2] = df3['value'].values[8000:10000]
#clients[3] = df4['glo_avg'].values[:8000]

drawnClient = np.random.randint(0,3)
testset = serverSet[drawnClient]

minMax = np.array([0, 1])
partitioner = Grid.GridPartitioner(data=minMax, npart=3, mf=mf.trimf)
                  
model = FCM_FTS.FCM_FTS(partitioner=partitioner, order=2, num_fcms=2,
                  activation_function=Activations.relu,
                  loss=lossFunction.func, param=True)

initialParameters = model.get_parameters()
print("=================== Initial Param ===========================")
print(initialParameters)

#print("FitRes:")
#for x in ClientProxy.get_parameters():
#    print(x)

class HRFTSStrategy(Strategy):
    
    # def __init__(self):
    #     super().__init__()
    
    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, FitRes]],
    #     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    # ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
    #     for _, fit_res in results:
    #         print(fit_res.parameters)
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            print("================= Results ===========================")
            for _, fit_res in results:
                print(parameters_to_ndarrays(fit_res.parameters))
                
            # Does in-place weighted average of results
            minResult = np.inf
            maxResult = -np.inf
            #aggResults = copy.deepcopy(results)
            #print("================== Results =====================")
            for i in range(len(results)):
            #    print("========= res ===============")
                minResult = min(parameters_to_ndarrays(results[i][1].parameters)[0], minResult)
                maxResult = max(parameters_to_ndarrays(results[i][1].parameters)[1], maxResult)
                aggParam = (parameters_to_ndarrays(results[i][1].parameters)[2:])
                #aggResults.append((results[i][0], ))
                aggDelete = parameters_to_ndarrays(results[i][1].parameters)
                del aggDelete[0:2]
                aggDeleteParam = ndarrays_to_parameters(aggDelete)
                results[i][1].parameters = aggDeleteParam
                #print(parameters_to_ndarrays(aggResults[i][1].parameters))
            
            #print(parameters_to_ndarrays(results[0][1]))
            #print(aggResults)
            #print(minResult)
            #print(maxResult)
            #print("==================== Results =======================")
            #for _, fit_res in results:
            #    print(parameters_to_ndarrays(fit_res.parameters))
            
            
            #print("=============================== Aggregated =========================")
            aggregated_ndarrays = aggregate_inplace(results)
            #print(aggregated_ndarrays)
            
            aggregated_ndarrays.insert(0, minResult)
            aggregated_ndarrays.insert(1, maxResult)
                
            #print(aggregated_ndarrays)

            #print("================= Results ===========================")
            #for _, fit_res in results:
            #    print(parameters_to_ndarrays(fit_res.parameters)[2:])
        else:
            #print("================= Results ===========================")
            #for _, fit_res in results:
            #    print(fit_res.num_examples[2:])
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
    
    
def aggregate_fit(
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        for _, fit_res in results:
            print(fit_res.parameters)
    

def get_evaluate_fn(testset: testset):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model.set_parameters(parameters)
        #print(model.get_parameters())
        forecasted = model.predict(testset)
        _rmse  = Measures.rmse(testset, forecasted, model.order-1)
        x = np.max(testset) - np.min(testset)
        nrmse = _rmse/x
        #rmse = model.evaluate(test)
        return nrmse, {"rmse": _rmse, "nrmse": nrmse}
    
    return evaluate

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    print("===================================== Metrics =======================================")
    print(metrics)
    print("===================================================================================")
    rmse = [num_examples * m["rmse"] for num_examples, m in metrics]
    nrmse = [num_examples * m["nrmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"rmse":sum(rmse) / sum(examples), "nrmse": sum(nrmse) / sum(examples)}


# Define strategy
strategy = HRFTSStrategy(initial_parameters=ndarrays_to_parameters(initialParameters))
#strategy = fl.server.strategy.FedAvg(
#    evaluate_metrics_aggregation_fn=weighted_average,
#    evaluate_fn=get_evaluate_fn(testset),  # global evaluation function
#    #initial_parameters=fl.common.ndarrays_to_parameters(model.get_parameters()),
#    #on_fit_config_fn=configure_fit_fn(),
#    
#    )

#print("Initial Parameters")
#print(parameters_to_ndarrays(strategy.initial_parameters))

# Start Flower server
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=numRounds),
    #strategy=fl.server.strategy.FedAvg(),
    strategy=strategy,
)

#print(getDataset.datasetDir)
#print(history)

print(f"{history.metrics_centralized = }")

#plotHistory.plot(history)