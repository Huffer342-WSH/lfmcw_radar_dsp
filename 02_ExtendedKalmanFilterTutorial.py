#!/usr/bin/env python

"""
=============================================
2 - Non-linear models: extended Kalman filter
=============================================
"""

# %%
# Nearly-constant velocity example
# --------------------------------
#
# We're going to use the same target model as previously, but this time we use a non-linear sensor
# model.
import numpy as np
from datetime import datetime, timedelta

start_time = datetime.now().replace(microsecond=0)

# %%

np.random.seed(1991)

# %%
# Create a target
# ^^^^^^^^^^^^^^^
#
# As in the previous tutorial, the target moves with near constant velocity NE from 0,0.
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05), ConstantVelocity(0.05)])
timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

for k in range(1, 21):
    timesteps.append(start_time + timedelta(seconds=k))
    truth.append(GroundTruthState(transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)), timestamp=timesteps[k]))

# %%
# Plot this

from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig

# %%
# A bearing-range sensor
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We're going to simulate a 2d bearing-range sensor. An example of this is available in Stone Soup
# via the :class:`~.CartesianToBearingRange` class. As the name suggests, this takes the Cartesian
# state input and returns a relative bearing and range to target. Specifically,
#
# .. math::
#       \begin{bmatrix}
#       \theta\\
#       r\\
#       \end{bmatrix}
#       =
#       \begin{bmatrix}
#       \arctan\left(\frac{y-y_p}{x-x_p}\right)\\
#       \sqrt{(x-x_p)^2 + (y-y_p)^2}
#       \end{bmatrix}
#
# where :math:`x_p,y_p` is the 2d Cartesian position of the sensor and :math:`x,y` that of the
# target. Note also that the arctan function has to resolve the quadrant ambiguity and so is
# implemented as the :class:`~.numpy.arctan2`:math:`(y,x)` function in Python.
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

sensor_x = 50  # Placing the sensor off-centre
sensor_y = 0

measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.2), 1]),  # Covariance matrix. 0.2 degree variance in
    # bearing and 1 metre in range
    translation_offset=np.array([[sensor_x], [sensor_y]]),  # Offset measurements to location of
    # sensor in cartesian.
)

# %%
# We create a set of detections using this sensor model.
from stonesoup.types.detection import Detection

measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp, measurement_model=measurement_model))

# %%
# Plot the measurements. Where the model is nonlinear the plotting function uses the inverse
# function to get coordinates.

plotter.plot_measurements(measurements, [0, 2])
plotter.fig

# %%
# Set up the extended Kalman filter elements
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Analogously to the Kalman filter, we now create the predictor and updater. As is our custom
# the predictor takes a transition model and the updater a measurement model. Note that if either
# of these models are linear then the extended predictor/updater defaults to its Kalman equivalent.
# In fact the extended Kalman filter classes inherit nearly all of their functionality from the
# Kalman classes. The only difference being that instead of returning a matrix, in the extended
# version the :meth:`~.MeasurementModel.matrix()` function returns the Jacobian.
from stonesoup.predictor.kalman import ExtendedKalmanPredictor

predictor = ExtendedKalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater

updater = ExtendedKalmanUpdater(measurement_model)

# %%
# Run the extended Kalman filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First, we'll create a prior state.
from stonesoup.types.state import GaussianState

prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# %%
# Next iterate over hypotheses and place in a track.
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# %%
# Plot the resulting track complete with error ellipses at each estimate.

plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig

# %%
# Key points
# ----------
# 1. Non-linear models can be incorporated into the tracking scheme through the use of the
#    appropriate combination of :class:`~.Predictor`/:class:`~.TransitionModel` or
#    :class:`~.Updater`/:class:`~.MeasurementModel`
# 2. Because Stone Soup uses inheritance, the amount of engineering required to achieve this is
#    minimal and the interface is the same. A user can apply the EKF components in exactly the same
#    way as the KF.

# %%
# The first order approximations used by the EKF provide a simple way to handle non-linear tracking
# problems. However, in highly non-linear systems these simplifications can lead to large errors in
# both the posterior state mean and covariance. In instances where we have noisy transition, or
# perhaps unreliable measurement, this could lead to a sub-optimal performance or even divergence
# of the filter. In the next tutorial, we see how the **unscented Kalman filter** can begin to
# address these issues.

# %%
# References
# ----------
# .. [#] Anderson & Moore 2012, Optimal filtering,
#        (http://users.cecs.anu.edu.au/~john/papers/BOOK/B02.PDF)

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/Tutorial_2.PNG'
