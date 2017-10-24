#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

enum RADAR_DATA_INDEX {
  RADAR_RHO_MEASURED,
  RADAR_PHI_MEASURED,
  RADAR_RHODOT_MEASURED
};

enum LIDAR_DATA_INDEX {
  LIDAR_X_MEASURED,
  LIDAR_Y_MEASURED,
};

typedef struct {
  VectorXd  x;
  MatrixXd  P;
} StateAndCovariance;

static MatrixXd CreateAugmentedSigmaPoints(const VectorXd& x, const MatrixXd& P, int n_aug, double std_a, double std_yawdd, double lambda);
static MatrixXd PredictSigmaPoints(const MatrixXd& Xsig_aug, double delta_t, int n_x, int n_aug);
static VectorXd PredictStateMean(int n_x, int n_aug, const VectorXd& weights, const MatrixXd& Xsig_pred);
static MatrixXd PredictStateCovariance(int n_x, int n_aug, const VectorXd& x, const VectorXd& weights, const MatrixXd& Xsig_pred);
static StateAndCovariance UpdateState(const VectorXd& z,
                                      const int n_x,
                                      const int n_aug,
                                      const MatrixXd Xsig_pred,
                                      const VectorXd weights,
                                      const std::function <VectorXd (const VectorXd&)> transform,
                                      const MatrixXd& R,
                                      const std::function <void (VectorXd&, VectorXd&)> ccHelper,
                                      const StateAndCovariance& inputState);

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  n_x_ = 5;
  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

  std_a_ = 0.2;

  //Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  //create vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    x_ = VectorXd::Zero(n_x_);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       Convert radar from polar to cartesian coordinates and initialize state.
       */
      float x = meas_package.raw_measurements_[RADAR_RHO_MEASURED] * cos(meas_package.raw_measurements_[RADAR_PHI_MEASURED]);
      float y = meas_package.raw_measurements_[RADAR_RHO_MEASURED] * sin(meas_package.raw_measurements_[RADAR_PHI_MEASURED]);

      x_ << x, y, 10, 0, .5;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[LIDAR_X_MEASURED], meas_package.raw_measurements_[LIDAR_Y_MEASURED], 10, 0, 0.5;
    }

    P_ << 0.2, 0, 0, 0, 0,
          0, 0.2, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, .1, 0,
          0, 0, 0, 0, .1;


    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < (n_aug_ * 2 + 1); ++i) {
      weights_(i) = 1 / (2 * (lambda_ + n_aug_));
    }

    previous_timestamp_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;

  Prediction(dt);

  switch (meas_package.sensor_type_) {
    case MeasurementPackage::RADAR:
      if (use_laser_) {
        UpdateRadar(meas_package);
      }
      break;

    case MeasurementPackage::LASER:
      if (use_radar_) {
        UpdateLidar(meas_package);
      }
      break;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug = ::CreateAugmentedSigmaPoints(x_, P_, n_aug_, std_a_, std_yawdd_, lambda_);
  Xsig_pred_ = ::PredictSigmaPoints(Xsig_aug, delta_t, n_x_, n_aug_);
  x_ = ::PredictStateMean(n_x_, n_aug_, weights_, Xsig_pred_);
  P_ = ::PredictStateCovariance(n_x_, n_aug_, x_, weights_, Xsig_pred_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;

  if (meas_package.raw_measurements_.size() != n_z) {
    cerr << "Lidar measurement doesn't contain right size data" << endl;
    return; // abort?
  }

  auto transformer = [=](const VectorXd& pred_vec) {
    VectorXd result = VectorXd(n_z);

    result(0) = pred_vec(0);
    result(1) = pred_vec(1);

    return result;
  };

  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
  0, std_laspy_ * std_laspy_;

  StateAndCovariance r = UpdateState(meas_package.raw_measurements_,
                                     n_x_,
                                     n_aug_,
                                     Xsig_pred_,
                                     weights_,
                                     transformer,
                                     R,
                                     NULL,
                                     { .x = x_, .P = P_});
  
  x_ = r.x;
  P_ = r.P;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
   //create matrix for sigma points in measurement space
  int n_z = 3;

  if (meas_package.raw_measurements_.size() != n_z) {
    cerr << "Radar measurement doesn't contain right size data" << endl;
    return; // abort?
  }

  auto transformer = [=](const VectorXd& pred_vec) {
    VectorXd result = VectorXd(n_z);

    const double x = pred_vec(0);
    const double y = pred_vec(1);
    const double v = pred_vec(2);
    const double yaw = pred_vec(3);

    const double dist = sqrt(x * x + y * y);

    result(0) = dist;
    result(1) = atan2(y, x);
    result(2) = (x * cos(yaw) * v + y * sin(yaw) * v) / dist;

    return result;
  };

  auto ccHelper = [](VectorXd& Xdiff, VectorXd& Zdiff) {
    while (Xdiff(3) > M_PI) Xdiff(3) -= 2 * M_PI;
    while (Xdiff(3) < -M_PI) Xdiff(3) += 2 * M_PI;

    while (Zdiff(1) > M_PI) Zdiff(1) -= 2 * M_PI;
    while (Zdiff(1) < -M_PI) Zdiff(1) += 2 * M_PI;
  };

  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0,
  0, std_radphi_ * std_radphi_, 0,
  0, 0, std_radrd_ * std_radrd_;

  StateAndCovariance r = UpdateState(meas_package.raw_measurements_,
                                     n_x_,
                                     n_aug_,
                                     Xsig_pred_,
                                     weights_,
                                     transformer,
                                     R,
                                     ccHelper,
                                     { .x = x_, .P = P_});

  x_ = r.x;
  P_ = r.P;
}

// ================================================================================
// PURE FUNCTIONS
// ================================================================================


static MatrixXd CreateAugmentedSigmaPoints(const VectorXd& x, const MatrixXd& P, int n_aug, double std_a, double std_yawdd, double lambda) {
  //create augmented mean state
  VectorXd x_aug = VectorXd::Zero(n_aug);
  x_aug.head(5) = x;

  //create augmented covariance matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug, n_aug);
  P_aug.topLeftCorner(5, 5) = P;
  P_aug(5, 5) = std_a * std_a;
  P_aug(6, 6) = std_yawdd * std_yawdd;

  //create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  //create augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  Xsig_aug.col(0) = x_aug;
  for (int i = 1; i <= n_aug; ++i) {
    Xsig_aug.col(i) = x_aug + sqrt(lambda + n_aug) * A_aug.col(i - 1);
    Xsig_aug.col(i + n_aug) = x_aug - sqrt(lambda + n_aug) * A_aug.col(i - 1);
  }

  return Xsig_aug;
}

static MatrixXd PredictSigmaPoints(const MatrixXd& Xsig_aug, double delta_t, int n_x, int n_aug) {
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  for (int i = 0; i < (2 * n_aug + 1); ++i) {
    const double px = Xsig_aug(0, i);
    const double py = Xsig_aug(1, i);
    const double v = Xsig_aug(2, i);
    const double psi = Xsig_aug(3, i);
    const double psid = Xsig_aug(4, i);
    const double nu_a = Xsig_aug(5, i);
    const double nu_dd = Xsig_aug(6, i);

    const double halfdt2 = 0.5 * (delta_t * delta_t);

    double px_p, py_p;

    if (fabs(psid) > .001) {
      px_p = px + v / psid * (sin(psi + psid * delta_t) - sin(psi));
      py_p = py + v / psid * (-cos(psi + psid * delta_t) + cos(psi));
    } else {
      px_p = px + v * cos(psi) * delta_t;
      py_p = py + v * sin(psi) * delta_t;
    }

    double v_p = v;
    double psi_p = psi + (psid * delta_t);
    double psid_p = psid;

    px_p += halfdt2 * cos(psi) * nu_a;
    py_p += halfdt2 * sin(psi) * nu_a;
    v_p += delta_t * nu_a;
    psi_p += halfdt2 * nu_dd;
    psid_p += delta_t * nu_dd;

    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = psi_p;
    Xsig_pred(4, i) = psid_p;
  }

  return Xsig_pred;
}

static VectorXd PredictStateMean(int n_x, int n_aug, const VectorXd& weights, const MatrixXd& Xsig_pred) {
  VectorXd x = VectorXd::Zero(n_x);
  for (int i = 0; i < (n_aug * 2 + 1); ++i) {
    x = x + weights(i) * Xsig_pred.col(i);
  }
  return x;
}

static MatrixXd PredictStateCovariance(int n_x, int n_aug, const VectorXd& x, const VectorXd& weights, const MatrixXd& Xsig_pred) {
  MatrixXd P = MatrixXd::Zero(n_x, n_x);
  for (int i = 0; i < (n_aug * 2 + 1); ++i) {
    VectorXd diff = Xsig_pred.col(i) - x;

    //angle normalization
    while (diff(3) > M_PI) diff(3) -= 2 * M_PI;
    while (diff(3) < -M_PI) diff(3) += 2 * M_PI;

    P = P + weights(i) * diff * diff.transpose();
  }
  return P;
}

static StateAndCovariance UpdateState(const VectorXd& z,
                                      const int n_x,
                                      const int n_aug,
                                      const MatrixXd Xsig_pred,
                                      const VectorXd weights,
                                      const std::function <VectorXd (const VectorXd&)> transform,
                                      const MatrixXd& R,
                                      const std::function <void (VectorXd&, VectorXd&)> ccHelper,
                                      const StateAndCovariance& inputState) {

  int n_z = z.size();

  // tranform the data in a form we want
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
  for (int i = 0; i < (2 * n_aug + 1); ++i) {
    Zsig.col(i) = transform(Xsig_pred.col(i));
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  for (int i = 0; i < (2 * n_aug + 1); ++i) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);

  for (int i = 0; i < (2 * n_aug + 1); ++i) {
    MatrixXd diff = Zsig.col(i) - z_pred;

    S = S + weights(i) * diff * diff.transpose();
  }

  S += R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x, n_z);

  for (int i = 0; i < 2 * n_aug + 1; i++) {
    VectorXd Xdiff = Xsig_pred.col(i) - inputState.x;
    VectorXd Zdiff = Zsig.col(i) - z_pred;

    if (ccHelper) {
      ccHelper(Xdiff, Zdiff);
    }

    Tc = Tc + weights(i) * Xdiff * Zdiff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd diff = z - z_pred;

  while (diff(1) > M_PI) diff(1) -= 2 * M_PI;
  while (diff(1) < -M_PI) diff(1) += 2 * M_PI;

  return {
    .x = inputState.x + K * diff,
    .P = inputState.P - K * S * K.transpose()
  };
}
