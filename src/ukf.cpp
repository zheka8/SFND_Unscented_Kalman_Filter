#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = false;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0; //30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6; //30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

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
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  is_initialized_ = false;
  
  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);

  // time when the state is true, in us
  time_us_ = 0;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_ + 1);

  // Sigma point spreading parameter
  lambda_ = 3.0 - n_x_;

  // set all matrices with initial values
  x_.fill(0.0);
  Xsig_pred_.fill(0.0);

  // set P to identitiy matrix
  P_.fill(0.0);
  for (int i=0; i < n_x_; ++i) {
    P_(i,i) = 1.0;
  }
    
  // set the weights vector
  double weight_0 = lambda_/ (lambda_ + n_aug_);
  double weight = 0.5 / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for(int i=1; i < 2*n_aug_ + 1; ++i) {
    weights_(i) = weight;
  }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_) {
    // initialize the filter with measurement values and don't predict
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {     
      x_.fill(0.0); 
      x_(0) = meas_package.raw_measurements_(0); // position
      x_(1) = meas_package.raw_measurements_(1); // velocity
      P_(0,0) = std_laspx_ * std_laspx_; // covariance x
      P_(1,1) = std_laspy_ * std_laspy_; // covariance y
      P_(2,2) = 100;
      P_(3,3) = 100;
      P_(4,4) = 10;
      std::cout<<"init LASER"<<x_<<std::endl;
      
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // convert coordinates from polar to cartesian
      double rho, phi, rhodot;
      rho = meas_package.raw_measurements_(0);
      phi = meas_package.raw_measurements_(1);
      rhodot = meas_package.raw_measurements_(2);

      // x and y coordinates
      x_(0) = rho * std::cos(phi);
      x_(1) = rho * std::sin(phi);

      double v_x = rhodot*std::cos(phi);
      double v_y = rhodot*std::sin(phi);

      double v  = sqrt(v_x*v_x + v_y*v_y);  

      // velocity
      //x_(2) = sqrt(rhodot * std::cos(phi) * rhodot * std::cos(phi) + rhodot * std::sin(phi) * rhodot * std::sin(phi));      
      x_(2) = v;
      std::cout<<"init RADAR"<<x_<<std::endl;

      
      P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
            0, std_radr_ * std_radr_, 0, 0, 0,
            0, 0, std_radrd_ * std_radrd_, 0, 0,
            0, 0, 0, std_radphi_ * std_radphi_, 0, 0,
            0, 0, 0, 0, std_radrd_ * std_radrd_;

      P_ << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0,
      0, 0, 0.5, 0, 0,
      0, 0, 0, 0.5, 0, 0,
      0, 0, 0, 0, 0.5;

    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

  } else {
    // predict and update

    // get the current and elapsed times
    double dt = (meas_package.timestamp_ - time_us_) / 1e6;
    time_us_ = meas_package.timestamp_;

    // predict using motion model
    Prediction(dt);

    // updated based on measurement type
    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) { 
      UpdateLidar(meas_package);
    } else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    }
  }
  //std::cout<<"Process Measurement End"<<std::endl;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // create augmented matrices
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);

  // populate augmented matrices
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0.0;      // mean accel noise
  x_aug(n_x_ + 1) = 0.0;  // mean yaw double dot rate noise
  std::cout<<"x aug"<<x_<<std::endl;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_;

  std::cout<<"P aug"<<P_aug<<std::endl;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
  std::cout<<"Xsig_aug_"<<Xsig_aug<<std::endl;

  // transform sigma points into measurement space
  for (int i = 0; i < 2*n_aug_ + 1; ++i) {
    // extract values for better readabiilty
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * (std::sin(yaw + yawd*delta_t) - std::sin(yaw));
      py_p = p_y + v/yawd * (std::cos(yaw) - cos(yaw + yawd*delta_t));
    } else {
      px_p = p_x + v*delta_t*std::cos(yaw);
      py_p = p_y + v*delta_t*std::sin(yaw);
      //std::cout <<"yawd < 0.001" << std::endl;
    }

    double v_p = v; //constant velocity
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;
    
    // add noise
    px_p += 0.5*nu_a*delta_t*delta_t * std::cos(yaw);
    py_p += 0.5*nu_a*delta_t*delta_t * std::sin(yaw);
    v_p += v_p + nu_a*delta_t;
    yaw_p += 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p += nu_yawdd*delta_t;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;

  }
  std::cout<<"Xsig_pred"<<Xsig_pred_<<std::endl;

  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }
  std::cout<<"x_"<<x_<<std::endl;

  // predicted state covariance
  //std::cout<<"predicted state covariance"<<std::endl;

  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3)+=2.*M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
  std::cout<<"P_"<<P_<<std::endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // get the measurment and it's size
  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size();
  
  // Measurement matrix
  MatrixXd H = MatrixXd(n_z, n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  // Measurement covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;

  VectorXd z_pred = VectorXd(n_z);
  z_pred = x_.head(n_z);

  VectorXd z_diff = z - z_pred;
  MatrixXd S = H * P_ * H.transpose() + R;
  MatrixXd K = P_ * H.transpose() * S.inverse();

  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);

  // Update the mean and covariance matrix
  x_ = x_ + (K * z_diff);
  P_ = (I - K * H) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // get the measurment and it's size
  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size();

  // sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);  // sigma points
  VectorXd z_pred = VectorXd(n_z);            // predicted mean
  MatrixXd S = MatrixXd(n_z, n_z);            // covariance

  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);

  // transform sig points to meas space
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    double p_x      = Xsig_pred_(0,i);
    double p_y      = Xsig_pred_(1,i);
    double v        = Xsig_pred_(2,i);
    double yaw      = Xsig_pred_(3,i);

    double v_x = v*std::cos(yaw);
    double v_y = v*std::sin(yaw);

    double rho  = sqrt(p_x*p_x + p_y*p_y);
    double phi  = atan2(p_y,p_x);
    double rhod = (p_x*v_x + p_y*v_y) / rho;
  
    Zsig(0,i) = rho;
    Zsig(1,i) = phi;
    Zsig(2,i) = rhod;
  }
  //std::cout<<Zsig<<std::endl;

  // predicted mean meas.
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }
  //std::cout<<z_pred<<std::endl;

  // innovation covariance matrix S
  for (int i=0; i < 2*n_aug_+1; ++i) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  //std::cout<<S<<std::endl;

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  
  S += R;
  
  //std::cout<<"cross correlation"<<std::endl;
  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for (int i=0; i < 2*n_aug_+1; ++i) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1)+=2.*M_PI;

    //state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3)+=2.*M_PI;
    //std::cout<<"before"<<std::endl;
    Tc += weights_(i) * x_diff * z_diff.transpose();
    //std::cout<<"after"<<std::endl;
  }
  //std::cout<<"radar Kalman gain"<<std::endl;
  // kalman gain
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;
  
  //angle normalization
  while (z_diff(1) >  M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1) < -M_PI) z_diff(1)+=2.*M_PI;

  // update state mean and covariance
  x_ += K * z_diff;
  P_ -= K*S*K.transpose();
  //std::cout<<x_<<std::endl;

}