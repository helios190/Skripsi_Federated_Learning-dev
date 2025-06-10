import flwr as fl
from keras.optimizers.schedules import ExponentialDecay   # Keras 3 schedule
from keras.optimizers           import Adam      # fast on M-series
from utils.utils import getDataset, get_model, evaluate_metrics
from sklearn.metrics import roc_auc_score
from utils.fedDFserverAdaptive import adaptive_clip_inputs_inplace
import logging

logging.basicConfig(level=logging.DEBUG)

x_train, y_train, x_test, y_test = getDataset(client_id=0, num_clients=2, split_ratios=[0.6, 0.4])
model, lr_schedule, earlystop = get_model()
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, earlystop):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.earlystop = earlystop

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=config.get("epochs", 1),
            batch_size=config.get("batch_size", 32),
            validation_data=(self.x_test, self.y_test),
            callbacks=[self.earlystop],
            verbose=1
        )

        updated_weights = self.model.get_weights()
        updates = [w - p for w, p in zip(updated_weights, parameters)]

        clipping_norm = config.get("clipping_norm", None)
        if clipping_norm is not None:
            updates = adaptive_clip_inputs_inplace(updates, clipping_norm)

        return [p + u for p, u in zip(parameters, updates)], len(self.x_train), {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1]
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        y_pred_probs = self.model.predict(self.x_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        acc, recall, precision, f1 = evaluate_metrics(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_probs)

        logging.info(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1: {f1}, AUC: {auc}")

        return loss, len(self.x_test), {
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "auc": auc
        }

client = FlwrClient(model, x_train, y_train, x_test, y_test, earlystop)
fl.client.start_numpy_client(server_address="localhost:8080", client=client)
