#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.size() == 0) {
    cout << "Estimation vector size cannot be zero";
    return rmse;
  }

  if (estimations.size() != ground_truth.size()) {
    cout << "Estimations and ground truth must be of the same size";
    return rmse;
  }

  VectorXd diff;

  for(int i=0; i < estimations.size(); ++i){
    diff = estimations[i] - ground_truth[i];
    diff = diff.array() * diff.array();
    rmse += diff;
  }

  rmse = rmse / estimations.size();

  rmse = rmse.array().sqrt();

  return rmse;
}
