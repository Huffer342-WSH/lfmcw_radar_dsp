import numpy as np
from .models import KalmanModel


class GaussianState:
    def __init__(self, state_vector, covar, timestamp):
        self.state_vector = state_vector
        self.covar = covar
        self.timestamp = timestamp


class GaussianMeasurementPrediction:
    def __init__(self, state_vector, covar, cross_covar, timestamp):
        self.state_vector = state_vector
        self.covar = covar
        self.cross_covar = cross_covar
        self.timestamp = timestamp


class Hypothesis:
    def __init__(self, prediction, measurement: np.ndarray, measurement_prediction=None):
        self.prediction = prediction
        self.measurement = measurement
        self.measurement_prediction = measurement_prediction


class Tracker:
    def __init__(self, state: GaussianState):
        self.state = state


class KalmanPredictor:
    def __init__(self, transition_model: KalmanModel):
        self.transition_model = transition_model

    def predict(self, state: GaussianState, timestamp):
        x_prior = state.state_vector
        p_prior = state.covar
        dt = timestamp - state.timestamp
        ff = self.transition_model.matirx(dt)
        qq = self.transition_model.covar(dt)
        x_pred = self.transition_model.function(x_prior, dt)

        p_pred = ff @ p_prior @ ff.T + qq
        return GaussianState(x_pred, p_pred, timestamp)


class KalmanUpdater:
    def __init__(self, measurement_model: KalmanModel):
        self.measurement_model = measurement_model

    def update(self, hypothesis: Hypothesis) -> GaussianState:
        if hypothesis.measurement_prediction is None:
            hypothesis.measurement_prediction = self.predict_measurement(hypothesis.prediction)

        z = hypothesis.measurement

        x_pred = hypothesis.prediction.state_vector
        p_pred = hypothesis.prediction.covar

        z_pred = hypothesis.measurement_prediction.state_vector
        s = hypothesis.measurement_prediction.covar
        ph = hypothesis.measurement_prediction.cross_covar

        hh = self.measurement_model.matrix(x_pred)

        rr = self.measurement_model.covar()

        k = ph @ np.linalg.inv(s)
        I_KH = np.identity(k.shape[0]) - k @ hh
        p_post = I_KH @ p_pred @ I_KH.T + k @ rr @ k.T

        x_post = x_pred + k @ (z - z_pred)
        return GaussianState(x_post, p_post, hypothesis.prediction.timestamp)

    def predict_measurement(self, predicted_state: GaussianState) -> GaussianMeasurementPrediction:
        hh = self.measurement_model.matrix(predicted_state.state_vector)
        rr = self.measurement_model.covar()
        state_vector = self.measurement_model.function(predicted_state.state_vector)
        cross_covar = predicted_state.covar @ hh.T
        covar = hh @ cross_covar + rr

        return GaussianMeasurementPrediction(state_vector, covar, cross_covar, predicted_state.timestamp)
