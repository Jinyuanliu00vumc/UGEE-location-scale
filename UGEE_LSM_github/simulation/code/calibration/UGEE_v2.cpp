// UGEE_v2.cpp
// Complete-data UGEE (no missing data).
// Identical to Dist_UGEE_v2.cpp except:
//   - accepts Beta_init / Eta_init from R instead of hardcoded starting values
//   - function renamed to ugee_solve_v2() to coexist with original if needed
//
// Parameter order:
//   Beta = (beta0, beta1, beta2)
//   Eta  = (alpha0, alpha1, tau0, tau1, tau2)
//
// Sandwich uses Hajek formula:  Sigma_U = (4/(n-1)) * sum_i v_i v_i^T
//                               Sigma_theta = B^{-1} Sigma_U B^{-1} / n

#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::depends(RcppEigen)]]

// Safe solve for tiny SPD systems
template <typename Mat, typename Vec>
inline Vec spd_solve(const Mat& A, const Vec& b) {
  Eigen::LLT<Mat> llt(A);
  if (llt.info() == Eigen::NumericalIssue) {
    Mat Ar = A + 1e-10 * Mat::Identity(A.rows(), A.cols());
    return Ar.ldlt().solve(b);
  }
  return llt.solve(b);
}

// Pseudoinverse for symmetric matrices
inline MatrixXd pinv_sym(const MatrixXd& A, double tol = 1e-10) {
  SelfAdjointEigenSolver<MatrixXd> es(A);
  if (es.info() != Success) {
    MatrixXd Ar = A + tol * MatrixXd::Identity(A.rows(), A.cols());
    return Ar.ldlt().solve(MatrixXd::Identity(A.rows(), A.cols()));
  }
  const VectorXd vals = es.eigenvalues();
  const MatrixXd V    = es.eigenvectors();
  VectorXd inv = vals.unaryExpr([&](double x){ return (x > tol) ? 1.0/x : 0.0; });
  return V * inv.asDiagonal() * V.transpose();
}

// [[Rcpp::export]]
Rcpp::List ugee_solve_v2(List           y_ki_all,
                         NumericVector  ui_R,
                         double         t_star,
                         NumericVector  Beta_init,   // length 3
                         NumericVector  Eta_init,    // length 5: (alpha0,alpha1,tau0,tau1,tau2)
                         double         eps      = 1e-5,
                         int            max_iter = 500) {
  // --------------------
  // Basic dimensions
  // --------------------
  const int K = y_ki_all.size();
  if (K < 2) stop("K must be >= 2");
  NumericVector y0 = y_ki_all[0];
  const int n = y0.size();
  if (n < 2) stop("n must be >= 2");

  if (Beta_init.size() != 3) stop("Beta_init must have length 3");
  if (Eta_init.size()  != 5) stop("Eta_init must have length 5");

  // Copy y into Eigen
  std::vector<VectorXd> y(K, VectorXd(n));
  for (int k = 0; k < K; ++k) {
    NumericVector yk = y_ki_all[k];
    if (yk.size() != n) stop("All y_ki_all[[k]] must have same length n.");
    for (int i = 0; i < n; ++i) y[k](i) = yk[i];
  }
  VectorXd ui(n);
  for (int i = 0; i < n; ++i) ui(i) = ui_R[i];

  const int n_comb = n * (n - 1) / 2;

  // --------------------
  // Parameter blocks — initialised from R
  // --------------------
  const int p_Beta = 3;
  const int p_Eta  = 5;   // (alpha0, alpha1, tau0, tau1, tau2)
  const int p      = p_Beta + p_Eta;

  VectorXd Beta(p_Beta), Eta(p_Eta);
  for (int i = 0; i < p_Beta; ++i) Beta(i) = Beta_init[i];
  for (int i = 0; i < p_Eta;  ++i) Eta(i)  = Eta_init[i];

  // k design (K x 3): [1, k, (k - t_star)+]
  MatrixXd k_vecs(K, 3);
  for (int k = 0; k < K; ++k) {
    const double kk  = static_cast<double>(k + 1);
    const double kpl = std::max(0.0, kk - t_star);
    k_vecs.row(k) << 1.0, kk, kpl;
  }

  // ----------------------------------------
  // STEP 1: estimate Beta from f1 only
  // ----------------------------------------
  int iter1 = 0;
  for (;;) {
    VectorXd U1  = VectorXd::Zero(p_Beta);
    MatrixXd Wr1 = MatrixXd::Zero(p_Beta, p_Beta);

    const double a0_ = Eta(0), a1_ = Eta(1);
    VectorXd sigb(n);
    for (int ii = 0; ii < n; ++ii) sigb(ii) = std::exp(a0_ + a1_ * ui(ii));

    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double h2_pair = 0.5 * (sigb(i) + sigb(j));

        VectorXd Si1(K), Var_f1(K);
        for (int k = 0; k < K; ++k) {
          const double f1  = 0.5 * (y[k](i) + y[k](j));
          const double h1  = k_vecs.row(k).dot(Beta);
          Si1(k)    = f1 - h1;
          const double WS_k = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
          Var_f1(k) = 0.25 * (2.0 * h2_pair) + 0.5 * WS_k;
        }

        for (int r = 0; r < p_Beta; ++r) {
          double acc = 0.0;
          for (int k = 0; k < K; ++k)
            acc += k_vecs(k, r) / std::max(Var_f1(k), 1e-12) * Si1(k);
          U1(r) += acc;
        }
        for (int r = 0; r < p_Beta; ++r)
          for (int c = 0; c < p_Beta; ++c) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
              acc += k_vecs(k, r) / std::max(Var_f1(k), 1e-12) * k_vecs(k, c);
            Wr1(r, c) += acc;
          }
      }
    }

    VectorXd Beta_new = Beta + spd_solve(Wr1, U1);
    if ((Beta_new - Beta).cwiseAbs().maxCoeff() < eps) { Beta = Beta_new; break; }
    Beta = Beta_new;
    if (++iter1 > max_iter) {
      Rcpp::warning("Mean-step (Beta) did not converge within max_iter.");
      break;
    }
  }

  // ----------------------------------------
  // STEP 2: estimate Eta=(alpha,tau) from f2, f3 with Beta fixed
  // ----------------------------------------
  int iter2 = 0;
  for (;;) {
    VectorXd U23  = VectorXd::Zero(p_Eta);
    MatrixXd Wr23 = MatrixXd::Zero(p_Eta, p_Eta);

    const double a0 = Eta(0), a1 = Eta(1);
    VectorXd sigb(n), ui_sigb(n);
    for (int ii = 0; ii < n; ++ii) {
      sigb(ii)    = std::exp(a0 + a1 * ui(ii));
      ui_sigb(ii) = ui(ii) * sigb(ii);
    }

    std::vector<VectorXd> e(K, VectorXd(n));
    for (int k = 0; k < K; ++k) {
      const double h1 = k_vecs.row(k).dot(Beta);
      for (int ii = 0; ii < n; ++ii) e[k](ii) = y[k](ii) - h1;
    }

    const int m = (K - 1) + K;

    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double h2_pair  = 0.5 * (sigb(i) + sigb(j));
        const double dh2_da0  = h2_pair;
        const double dh2_da1  = 0.5 * (ui_sigb(i) + ui_sigb(j));

        VectorXd Si23(m), Var23(m);
        MatrixXd Gi23(m, p_Eta); Gi23.setZero();

        // f2 block (K-1 rows)
        for (int k = 0; k < K - 1; ++k) {
          const double d1 = e[k](i)   - e[k](j);
          const double d2 = e[k+1](i) - e[k+1](j);
          Si23(k) = 0.5 * d1 * d2 - h2_pair;

          const double WS_k  = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
          const double WS_k1 = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k+1)));
          const double sD2   = 2.0 * h2_pair;
          Var23(k) = 0.25 * ((sD2 + 2.0*WS_k) * (sD2 + 2.0*WS_k1) + sD2*sD2);

          Gi23(k, 0) = dh2_da0;
          Gi23(k, 1) = dh2_da1;
          // tau cols = 0 for f2
        }

        // f3 block (K rows)
        for (int k = 0; k < K; ++k) {
          const double d   = e[k](i) - e[k](j);
          const double WS_k = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
          Si23((K-1)+k) = 0.5 * d*d - (h2_pair + WS_k);

          const double sD2  = 2.0 * h2_pair;
          const double v_k2 = sD2 + 2.0*WS_k;
          Var23((K-1)+k) = 0.5 * v_k2 * v_k2;

          Gi23((K-1)+k, 0) = dh2_da0;
          Gi23((K-1)+k, 1) = dh2_da1;
          Gi23((K-1)+k, 2) = k_vecs(k, 0) * WS_k;  // dtau0
          Gi23((K-1)+k, 3) = k_vecs(k, 1) * WS_k;  // dtau1
          Gi23((K-1)+k, 4) = k_vecs(k, 2) * WS_k;  // dtau2
        }

        for (int r = 0; r < p_Eta; ++r) {
          double accU = 0.0;
          for (int t = 0; t < m; ++t)
            accU += Gi23(t, r) / std::max(Var23(t), 1e-12) * Si23(t);
          U23(r) += accU;
        }
        for (int r = 0; r < p_Eta; ++r)
          for (int c = 0; c < p_Eta; ++c) {
            double accW = 0.0;
            for (int t = 0; t < m; ++t)
              accW += Gi23(t, r) / std::max(Var23(t), 1e-12) * Gi23(t, c);
            Wr23(r, c) += accW;
          }
      }
    }

    VectorXd Eta_new = Eta + spd_solve(Wr23, U23);
    if ((Eta_new - Eta).cwiseAbs().maxCoeff() < eps) { Eta = Eta_new; break; }
    Eta = Eta_new;
    if (++iter2 > max_iter) {
      Rcpp::warning("Variance-step (Eta) did not converge within max_iter.");
      break;
    }
  }

  // ============================================================
  // Sandwich variance (Hajek U-statistic)
  // ============================================================

  // --- Final Jacobian B ---
  MatrixXd Wr1_fin  = MatrixXd::Zero(p_Beta, p_Beta);
  MatrixXd Wr23_fin = MatrixXd::Zero(p_Eta,  p_Eta);

  const double a0f = Eta(0), a1f = Eta(1);
  VectorXd sigb_f(n), ui_sigb_f(n);
  for (int ii = 0; ii < n; ++ii) {
    sigb_f(ii)    = std::exp(a0f + a1f * ui(ii));
    ui_sigb_f(ii) = ui(ii) * sigb_f(ii);
  }

  std::vector<VectorXd> e_fin(K, VectorXd(n));
  for (int k = 0; k < K; ++k) {
    const double h1 = k_vecs.row(k).dot(Beta);
    for (int ii = 0; ii < n; ++ii) e_fin[k](ii) = y[k](ii) - h1;
  }

  const int m = (K - 1) + K;

  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double h2_pair = 0.5 * (sigb_f(i) + sigb_f(j));
      const double dh2_da0 = h2_pair;
      const double dh2_da1 = 0.5 * (ui_sigb_f(i) + ui_sigb_f(j));

      // Wr1_fin
      for (int r = 0; r < p_Beta; ++r)
        for (int c = 0; c < p_Beta; ++c) {
          double acc = 0.0;
          for (int k = 0; k < K; ++k) {
            const double WS_k   = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
            const double Var_f1 = 0.25 * (2.0*h2_pair) + 0.5*WS_k;
            acc += k_vecs(k, r) / std::max(Var_f1, 1e-12) * k_vecs(k, c);
          }
          Wr1_fin(r, c) += acc;
        }

      // Wr23_fin
      MatrixXd Gi23(m, p_Eta); Gi23.setZero();
      VectorXd Var23(m);

      for (int k = 0; k < K - 1; ++k) {
        const double WS_k  = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        const double WS_k1 = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k+1)));
        const double sD2   = 2.0 * h2_pair;
        Var23(k) = 0.25 * ((sD2+2.0*WS_k)*(sD2+2.0*WS_k1) + sD2*sD2);
        Gi23(k, 0) = dh2_da0;
        Gi23(k, 1) = dh2_da1;
      }
      for (int k = 0; k < K; ++k) {
        const double WS_k  = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        const double sD2   = 2.0 * h2_pair;
        const double v_k2  = sD2 + 2.0*WS_k;
        Var23((K-1)+k) = 0.5 * v_k2 * v_k2;
        Gi23((K-1)+k, 0) = dh2_da0;
        Gi23((K-1)+k, 1) = dh2_da1;
        Gi23((K-1)+k, 2) = k_vecs(k, 0) * WS_k;
        Gi23((K-1)+k, 3) = k_vecs(k, 1) * WS_k;
        Gi23((K-1)+k, 4) = k_vecs(k, 2) * WS_k;
      }
      for (int r = 0; r < p_Eta; ++r)
        for (int c = 0; c < p_Eta; ++c) {
          double accW = 0.0;
          for (int t = 0; t < m; ++t)
            accW += Gi23(t, r) / std::max(Var23(t), 1e-12) * Gi23(t, c);
          Wr23_fin(r, c) += accW;
        }
    }
  }

  MatrixXd B = MatrixXd::Zero(p, p);
  B.block(0,      0,      p_Beta, p_Beta) = Wr1_fin;
  B.block(p_Beta, p_Beta, p_Eta,  p_Eta)  = Wr23_fin;
  B /= static_cast<double>(n_comb);

  // --- Hajek influence functions ---
  MatrixXd Sigma_U = MatrixXd::Zero(p, p);

  for (int i = 0; i < n; ++i) {
    VectorXd v_i = VectorXd::Zero(p);

    for (int j = 0; j < n; ++j) {
      if (j == i) continue;
      const int i1 = std::min(i, j);
      const int i2 = std::max(i, j);

      const double h2_pair = 0.5 * (sigb_f(i1) + sigb_f(i2));
      const double dh2_da0 = h2_pair;
      const double dh2_da1 = 0.5 * (ui_sigb_f(i1) + ui_sigb_f(i2));

      // Beta contribution
      VectorXd Si1(K), Var_f1(K);
      for (int k = 0; k < K; ++k) {
        const double f1  = 0.5 * (y[k](i1) + y[k](i2));
        const double h1  = k_vecs.row(k).dot(Beta);
        Si1(k) = f1 - h1;
        const double WS_k = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        Var_f1(k) = 0.25 * (2.0*h2_pair) + 0.5*WS_k;
      }
      VectorXd vi_pair = VectorXd::Zero(p);
      for (int r = 0; r < p_Beta; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
          acc += k_vecs(k, r) / std::max(Var_f1(k), 1e-12) * Si1(k);
        vi_pair(r) = acc;
      }

      // Eta contribution
      VectorXd Si23(m), Var23(m);
      MatrixXd Gi23(m, p_Eta); Gi23.setZero();

      for (int k = 0; k < K - 1; ++k) {
        const double d1 = e_fin[k](i1)   - e_fin[k](i2);
        const double d2 = e_fin[k+1](i1) - e_fin[k+1](i2);
        Si23(k) = 0.5 * d1*d2 - h2_pair;
        const double WS_k  = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        const double WS_k1 = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k+1)));
        const double sD2   = 2.0 * h2_pair;
        Var23(k) = 0.25 * ((sD2+2.0*WS_k)*(sD2+2.0*WS_k1) + sD2*sD2);
        Gi23(k, 0) = dh2_da0;
        Gi23(k, 1) = dh2_da1;
      }
      for (int k = 0; k < K; ++k) {
        const double d    = e_fin[k](i1) - e_fin[k](i2);
        const double WS_k = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        Si23((K-1)+k) = 0.5*d*d - (h2_pair + WS_k);
        const double sD2  = 2.0 * h2_pair;
        const double v_k2 = sD2 + 2.0*WS_k;
        Var23((K-1)+k) = 0.5 * v_k2 * v_k2;
        Gi23((K-1)+k, 0) = dh2_da0;
        Gi23((K-1)+k, 1) = dh2_da1;
        Gi23((K-1)+k, 2) = k_vecs(k, 0) * WS_k;
        Gi23((K-1)+k, 3) = k_vecs(k, 1) * WS_k;
        Gi23((K-1)+k, 4) = k_vecs(k, 2) * WS_k;
      }
      for (int r = 0; r < p_Eta; ++r) {
        double acc = 0.0;
        for (int t = 0; t < m; ++t)
          acc += Gi23(t, r) / std::max(Var23(t), 1e-12) * Si23(t);
        vi_pair(p_Beta + r) = acc;
      }

      v_i.noalias() += vi_pair / static_cast<double>(n - 1);
    }

    // Sigma_U accumulation: 4/(n-1) * v_i v_i^T  (matches reference Dist_UGEE_v2.cpp)
    Sigma_U.noalias() += 4.0 * (v_i * v_i.transpose()) / static_cast<double>(n - 1);
  }

  // Sigma_theta = B^{+} * Sigma_U * B^{+} / n
  const MatrixXd Bpinv       = pinv_sym(B);
  const MatrixXd Sigma_theta = Bpinv * Sigma_U * Bpinv / static_cast<double>(n);
  const VectorXd sigma_theta = Sigma_theta.diagonal();

  // Pack outputs
  NumericVector theta_out(p);
  for (int i = 0; i < p_Beta; ++i) theta_out[i]         = Beta(i);
  for (int i = 0; i < p_Eta;  ++i) theta_out[p_Beta + i] = Eta(i);

  return List::create(
    _["theta.est"]  = theta_out,
    _["Beta"]       = Beta,
    _["Eta"]        = Eta,
    _["B"]          = B,
    _["Sigma.U"]    = Sigma_U,
    _["Sigma.est"]  = Sigma_theta,
    _["sigma.est"]  = sigma_theta,
    _["iter_beta"]  = iter1,
    _["iter_eta"]   = iter2,
    _["converged"]  = (iter1 <= max_iter && iter2 <= max_iter)
  );
}
