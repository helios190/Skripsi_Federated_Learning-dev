import numpy as np
import flwr as fl
from keras.optimizers.schedules import ExponentialDecay   # Keras 3 schedule
from keras.optimizers           import Adam      # fast on M-series
from utils.utils import getDataset, get_model, evaluate_metrics
import pandas as pd
from flwr.server.strategy import FedAvg
from utils.fedDFserverAdaptive import DifferentialPrivacyServerSideAdaptiveClipping 
from flwr.common import differential_privacy
from sklearn.metrics import roc_auc_score
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset for global evaluation (assuming client_id = 0, num_clients = 2)
x_train, y_train, x_test, y_test = getDataset(client_id=0, num_clients=2, split_ratios=[0.6, 0.4])

# Compile the model
model, lr_schedule, earlystop = get_model()
model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Global list to store metrics across rounds
evaluation_metrics = []
info_logs = []

# Define the evaluation function
def get_eval_fn(model, num_clients):
    def evaluate(server_round, parameters, config):
        model.set_weights(parameters)

        # Collect test data from all clients
        x_test_list, y_test_list = [], []
        for client_id in range(num_clients):
            _, _, x_test, y_test = getDataset(client_id, num_clients)
            x_test_list.append(x_test)
            y_test_list.append(y_test)

        x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)

        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        y_pred_probs = model.predict(x_test)
        y_pred = (y_pred_probs > 0.5).astype(int)
        acc, recall, precision, f1 = evaluate_metrics(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_probs)

        # Create metrics dictionary
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "auc": auc,
            "num_examples": len(y_test)
        }

        # Log metrics
        logging.info(f"Round {server_round} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        evaluation_metrics.append({"round": server_round, **metrics})

        return loss, metrics

    return evaluate

# Define a custom evaluate_metrics_aggregation_fn
def evaluate_metrics_aggregation_fn(metrics):
    aggregated_metrics = {}
    total_examples = sum(metric[0] for metric in metrics)

    for key in metrics[0][1].keys():
        aggregated_metrics[key] = sum(
            metric[1][key] * metric[0] / total_examples for metric in metrics
        )

    return aggregated_metrics

# Save metrics and logs to CSV
def save_metrics_to_csv(metrics, logs, metrics_filename="./results/FDFSAC/FDFSAC_0.01_50.csv", logs_filename="info_logs.csv"):
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_filename, index=False)
    logging.info(f"Metrics saved to {metrics_filename}")

    logs_df = pd.DataFrame({"logs": logs})
    logs_df.to_csv(logs_filename, index=False)
    logging.info(f"Logs saved to {logs_filename}")

# Instantiate the default DP strategy with hooks for logging clipped value and noise
dp_strategy = DifferentialPrivacyServerSideAdaptiveClipping(
    strategy=FedAvg(
        evaluate_fn=get_eval_fn(model, num_clients=2),
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    ),
    noise_multiplier=0.01,  # Adjusted for better privacy-utility trade-off
    num_sampled_clients=2,  # Fixed
    initial_clipping_norm=0.1,  # Start small for exponential growth
    target_clipped_quantile=0.9,  # Median clipping
    clip_norm_lr=0.2,  # Suggested learning rate for clipping norm
    clipped_count_stddev=0.1,  # Based on number of sampled clients
)

# Hook into the strategy to log clipping norm, noise, and NSR
original_aggregate_fit = dp_strategy.aggregate_fit

def hooked_aggregate_fit(server_round, results, failures):
    # Call the original method
    aggregated_params, metrics = original_aggregate_fit(server_round, results, failures)

    # Retrieve clipping statistics
    clipping_norm = dp_strategy.clipping_norm
    dp_noise_stddev = dp_strategy.noise_multiplier * clipping_norm
    privacy_budget = dp_strategy.noise_multiplier * (server_round ** 0.5)
    nsr = dp_noise_stddev / clipping_norm

    logging.info(f"Round {server_round} - Clipping norm: {clipping_norm:.4f}, Noise stdev: {dp_noise_stddev:.4f}, Privacy budget: {privacy_budget:.4f}, NSR: {nsr:.4f}")

    # Append these statistics to evaluation metrics
    evaluation_metrics.append({
        "round": server_round,
        "clipping_norm": clipping_norm,
        "dp_noise_stddev": dp_noise_stddev,
        "privacy_budget": privacy_budget,
        "nsr": nsr
    })

    return aggregated_params, metrics

# Replace the original method with the hooked one
dp_strategy.aggregate_fit = hooked_aggregate_fit

# Start federated server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=dp_strategy
)

# Save metrics to CSV after training
save_metrics_to_csv(evaluation_metrics, info_logs)
