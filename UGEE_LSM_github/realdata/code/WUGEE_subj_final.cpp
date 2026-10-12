// WUGEE_subj_v.cpp  (Model C — visit scale)
// ============================================================
// Weighted UGEE — Subject-level Missing Model
// Visit-number axis (K = 1..10), CHTN episodes only
//
// MODEL SPECIFICATION (Model C — pair-level absolute distances)
// -----------------------------------------------
// Mean model  (p_Beta = 7):
//   x[i][k] = [1, k, k^2, age_c_i, wt_mean_i, wt_dev_{i,k}, ckd_i]^T
//   E[y_{ik}] = x[i][k]^T Beta
//   (k = visit number 1..K, zero-indexed loop k=0..K-1 → local kk=k+1)
//
// Between-subject variance  (pair-level, time-invariant, p_Alpha = 5):
//   phi_ij = [1,
//             (age_i  - age_j )^2,
//             (wm_i   - wm_j  )^2,
//             I(|ckd_i - ckd_j| == 1),   <- discordant
//             ckd_i * ckd_j              ]   <- both CKD
//   h2_pair_ij = exp(phi_ij^T alpha)
//   sigD2_ij   = 2 * h2_pair_ij  =  Var(b_i - b_j)
//
// Within-subject variance  (pair+time-varying, p_Tau = 4):
//   w_{i,k} = [1,  k,  k^2,  wt_dev_{i,k}]
//   WS_{ij,k}  = exp(w_{i,k}^T tau) + exp(w_{j,k}^T tau)
//              = sigma^2_{i,k} + sigma^2_{j,k}
//   v_{ij,k}   = sigD2_ij + WS_{ij,k}  = Var(y_{ik} - y_{jk})
//
// Gradient of WS w.r.t. tau:
//   dWS/dtau = w_{i,k} * exp(w_{i,k}^T tau)  +  w_{j,k} * exp(w_{j,k}^T tau)
//
// Eta = [alpha (5), tau (4)]    (p_Eta = 9)
//
// Subject-level dropout model  (p_Xi = 2*(K-1)):
//   logit(p_{it}) = xi_{0t} + xi_{1t} * y_{i,t-1}
//   pi_{ik} = prod_{t=2}^{k} p_{it}(xi)
//
// Additive IPW for f^(1) (all pairs, even partially observed):
//   f^(1,IPW)_{ij,k} = 0.5*(r_{ik}*y_{ik}/pi_{ik} + r_{jk}*y_{jk}/pi_{jk})
//   V0_f1 = 0.25 * v_{ij,k}
//
// f^(2) and f^(3) use factorised subject-level weights (JOINT obs required):
//   lambda^(2)_{ij,k} = r_{i,k+1}*r_{j,k+1} / (pi_{i,k+1}*pi_{j,k+1})
//   lambda^(3)_{ij,k} = r_{ik}*r_{jk}        / (pi_{ik}*pi_{jk})
//   V0_f2 = 0.25*(v_k * v_{k+1} + sigD2^2)
//   V0_f3 = 0.5 * v_k^2
//
// INPUTS:
//   y_ki_all      : K-list of length-n outcomes (NA at dropout visits)
//   r_indiv_all   : K-list of length-n indicators r_{ik} in {0,1}
//   age_c_R       : length-n centred age
//   wt_mean_R     : length-n episode-mean weight (centred by grand mean)
//   wt_dev_ki_all : K-list of length-n weight deviations (weight - ep mean)
//   ckd_R         : length-n CKD indicator (0/1)
//   Beta_init (7), Eta_init (9), Xi_init (2*(K-1))
//   eps, max_iter
//
// OUTPUTS (List):
//   theta.est, Beta, Eta, Xi
//   Sigma.theta, sigma.theta  (Hajek for Beta+Eta)
//   Sigma.xi.logit, sigma.xi.logit
//   iter, converged
// ============================================================

// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
using namespace Rcpp;
using namespace Eigen;

// -------------------------------------------------------
// Helper: safe SPD solve with LDLT fallback
// -------------------------------------------------------
template <typename Mat, typename Vec>
inline Vec spd_solve(const Mat& A, const Vec& b) {
  Eigen::LLT<Mat> llt(A);
  if (llt.info() == Eigen::NumericalIssue) {
    Mat Ar = A + 1e-10 * Mat::Identity(A.rows(), A.cols());
    return Ar.ldlt().solve(b);
  }
  return llt.solve(b);
}

// -------------------------------------------------------
// Helper: pseudo-inverse for symmetric matrix via eigen
// -------------------------------------------------------
inline MatrixXd pinv_sym(const MatrixXd& A, double tol = 1e-10) {
  SelfAdjointEigenSolver<MatrixXd> es(A);
  if (es.info() != Eigen::Success) {
    MatrixXd Ar = A + tol * MatrixXd::Identity(A.rows(), A.cols());
    return Ar.ldlt().solve(MatrixXd::Identity(A.rows(), A.cols()));
  }
  const VectorXd vals = es.eigenvalues();
  const MatrixXd V    = es.eigenvectors();
  VectorXd inv = vals.unaryExpr(
    [&](double x){ return (x > tol) ? 1.0 / x : 0.0; });
  return V * inv.asDiagonal() * V.transpose();
}

// ============================================================
// Main exported function
// ============================================================
// [[Rcpp::export]]
Rcpp::List wugee_subj_v(
    Rcpp::List          y_ki_all,
    Rcpp::List          r_indiv_all,          // K-list of length-n r_{ik}
    Rcpp::NumericVector age_c_R,              // length-n centred age
    Rcpp::NumericVector wt_mean_R,            // length-n episode-mean weight
    Rcpp::List          wt_dev_ki_all,        // K-list of length-n wt deviations
    Rcpp::NumericVector ckd_R,               // length-n CKD indicator (0/1)
    Rcpp::NumericVector Beta_init,            // length 7
    Rcpp::NumericVector Eta_init,             // length 9
    Rcpp::NumericVector Xi_init,              // length 2*(K-1)
    double eps      = 1e-5,
    int    max_iter = 500
) {
  // --------------------------------------------------------
  // 0. Dimensions & validation
  // --------------------------------------------------------
  const int K = y_ki_all.size();
  if (K < 2) stop("K must be >= 2.");
  if ((int)r_indiv_all.size()    != K) stop("r_indiv_all must have length K.");
  if ((int)wt_dev_ki_all.size()  != K) stop("wt_dev_ki_all must have length K.");

  Rcpp::NumericVector y0 = y_ki_all[0];
  const int n = y0.size();
  if (n < 2) stop("n must be >= 2.");

  const int p_Beta  = 7;
  const int p_Alpha = 5;
  const int p_Tau   = 4;
  const int p_Eta   = p_Alpha + p_Tau;   // = 9
  const int p_Xi    = 2 * (K - 1);
  const int p_BE    = p_Beta + p_Eta;    // = 16
  const int p       = p_BE + p_Xi;

  if ((int)age_c_R.size()   != n) stop("age_c_R must have length n.");
  if ((int)wt_mean_R.size() != n) stop("wt_mean_R must have length n.");
  if ((int)ckd_R.size()     != n) stop("ckd_R must have length n.");
  if ((int)Beta_init.size() != p_Beta) stop("Beta_init must have length 7.");
  if ((int)Eta_init.size()  != p_Eta)  stop("Eta_init must have length 9.");
  if ((int)Xi_init.size()   != p_Xi)   stop("Xi_init must have length 2*(K-1).");

  // --------------------------------------------------------
  // 1. Copy inputs into Eigen
  // --------------------------------------------------------
  std::vector<VectorXd> y(K, VectorXd(n));
  for (int k = 0; k < K; ++k) {
    Rcpp::NumericVector yk = y_ki_all[k];
    if ((int)yk.size() != n) stop("All y_ki_all[[k]] must have length n.");
    for (int i = 0; i < n; ++i)
      y[k](i) = yk[i];
  }

  std::vector<VectorXd> r(K, VectorXd(n));
  for (int k = 0; k < K; ++k) {
    Rcpp::NumericVector rk = r_indiv_all[k];
    if ((int)rk.size() != n) stop("All r_indiv_all[[k]] must have length n.");
    for (int i = 0; i < n; ++i) r[k](i) = rk[i];
  }

  VectorXd age_c(n), wt_mean(n), ckd_v(n);
  for (int i = 0; i < n; ++i) {
    age_c(i)   = age_c_R[i];
    wt_mean(i) = wt_mean_R[i];
    ckd_v(i)   = ckd_R[i];
  }

  std::vector<VectorXd> wt_dev(K, VectorXd(n));
  for (int k = 0; k < K; ++k) {
    Rcpp::NumericVector dk = wt_dev_ki_all[k];
    if ((int)dk.size() != n) stop("All wt_dev_ki_all[[k]] must have length n.");
    for (int i = 0; i < n; ++i) wt_dev[k](i) = dk[i];
  }

  const int n_pairs = n * (n - 1) / 2;

  // --------------------------------------------------------
  // 2. Precompute subject design matrices x_sub[k].row(i)
  //    x[i][k] = [1, kk, kk^2, age_c(i), wt_mean(i), wt_dev[k](i), ckd(i)]
  //    (0-indexed k; paper's local time kk = k+1 = 1..K)
  // --------------------------------------------------------
  std::vector<MatrixXd> x_sub(K, MatrixXd(n, p_Beta));
  for (int k = 0; k < K; ++k) {
    const double kk = static_cast<double>(k + 1);
    for (int i = 0; i < n; ++i) {
      x_sub[k](i, 0) = 1.0;
      x_sub[k](i, 1) = kk;
      x_sub[k](i, 2) = kk * kk;
      x_sub[k](i, 3) = age_c(i);
      x_sub[k](i, 4) = wt_mean(i);
      x_sub[k](i, 5) = wt_dev[k](i);
      x_sub[k](i, 6) = ckd_v(i);
    }
  }

  // --------------------------------------------------------
  // 3. Initialise parameters
  // --------------------------------------------------------
  VectorXd Beta(p_Beta), Eta(p_Eta), Xi(p_Xi);
  for (int r_ = 0; r_ < p_Beta; ++r_) Beta(r_) = Beta_init[r_];
  for (int r_ = 0; r_ < p_Eta;  ++r_) Eta(r_)  = Eta_init[r_];
  for (int r_ = 0; r_ < p_Xi;   ++r_) Xi(r_)   = Xi_init[r_];

  // --------------------------------------------------------
  // 4. Working buffers
  // --------------------------------------------------------
  const double tiny = 1e-10;

  std::vector<VectorXd> pi_mat(K, VectorXd::Ones(n));
  std::vector<VectorXd> lam_i(K, VectorXd::Zero(n));
  std::vector<VectorXd> e(K, VectorXd(n));

  // --------------------------------------------------------
  // Lambda: refresh residuals given current Beta
  // --------------------------------------------------------
  auto refresh_resid = [&]() {
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < n; ++i)
        e[k](i) = y[k](i) - x_sub[k].row(i).dot(Beta);
  };

  // --------------------------------------------------------
  // Lambda: refresh pi and lam_i given current Xi
  // --------------------------------------------------------
  auto refresh_weights = [&]() {
    for (int k = 0; k < K; ++k) {
      pi_mat[k].setOnes();
      lam_i[k].setZero();
    }
    for (int i = 0; i < n; ++i)
      lam_i[0](i) = r[0](i);

    for (int i = 0; i < n; ++i) {
      double pi_cur = 1.0;
      for (int t = 2; t <= K; ++t) {
        const int k_idx = t - 1;
        if (r[t - 2](i) == 1.0) {
          Eigen::Vector2d g_t;
          g_t(0) = 1.0;
          g_t(1) = y[t - 2](i);
          const int off = 2 * (t - 2);
          const Eigen::Vector2d xi_t = Xi.segment<2>(off);
          const double p_t = 1.0 / (1.0 + std::exp(-g_t.dot(xi_t)));
          pi_cur *= p_t;
        }
        pi_mat[k_idx](i) = pi_cur;
        lam_i[k_idx](i)  = (pi_cur > tiny) ? r[k_idx](i) / pi_cur : 0.0;
      }
    }
  };

  // --------------------------------------------------------
  // Helper: build phi_ij (5-vector), h2_pair, sigD2
  // --------------------------------------------------------
  auto make_phi = [&](int i, int j, VectorXd& phi, double& h2, double& sigD2,
                      VectorXd& G_alpha, const VectorXd& alpha) {
    const double dag_abs  = std::abs(age_c(i)   - age_c(j));
    const double dwm_abs  = std::abs(wt_mean(i) - wt_mean(j));
    const double ckd_disc = std::abs(ckd_v(i) - ckd_v(j));
    const double ckd_both = ckd_v(i) * ckd_v(j);
    phi.resize(p_Alpha);
    phi << 1.0, dag_abs, dwm_abs, ckd_disc, ckd_both;
    h2      = std::exp(alpha.dot(phi));
    sigD2   = 2.0 * h2;
    G_alpha = phi * h2;
  };

  // Helper: build w_i_k (4-vector) and scalar ws_i = exp(w^T tau)
  auto ws_val = [&](int i, int k, const VectorXd& tau) -> double {
    const double kk = static_cast<double>(k + 1);
    Eigen::Vector4d w;
    w << 1.0, kk, kk * kk, wt_dev[k](i);
    return std::exp(tau.dot(w));
  };

  auto make_w = [&](int i, int k) -> Eigen::Vector4d {
    const double kk = static_cast<double>(k + 1);
    Eigen::Vector4d w;
    w << 1.0, kk, kk * kk, wt_dev[k](i);
    return w;
  };

  // --------------------------------------------------------
  // 5. Outer iteration
  // --------------------------------------------------------
  bool converged = false;
  int  iter      = 0;

  while (!converged && iter < max_iter) {
    ++iter;

    VectorXd Beta_old = Beta;
    VectorXd Eta_old  = Eta;
    VectorXd Xi_old   = Xi;

    // ====================================================
    // M-STEP: Newton for Xi (subject-level logistic)
    // ====================================================
    {
      VectorXd U_xi = VectorXd::Zero(p_Xi);
      MatrixXd B_xi = MatrixXd::Zero(p_Xi, p_Xi);

      for (int t = 2; t <= K; ++t) {
        const int off = 2 * (t - 2);
        Eigen::Vector2d U_t = Eigen::Vector2d::Zero();
        Eigen::Matrix2d B_t = Eigen::Matrix2d::Zero();

        for (int i = 0; i < n; ++i) {
          if (r[t - 2](i) != 1.0) continue;
          const double r_cur  = r[t - 1](i);
          const double y_prev = y[t - 2](i);
          Eigen::Vector2d g_t;
          g_t(0) = 1.0; g_t(1) = y_prev;
          const Eigen::Vector2d xi_t = Xi.segment<2>(off);
          const double p_t = 1.0 / (1.0 + std::exp(-g_t.dot(xi_t)));
          U_t.noalias() += g_t * (r_cur - p_t);
          B_t.noalias() += g_t * g_t.transpose() * p_t * (1.0 - p_t);
        }

        U_xi.segment<2>(off) = U_t;
        B_xi.block<2, 2>(off, off) = B_t;
      }
      for (int d = 0; d < p_Xi; ++d) B_xi(d, d) += 1e-8;
      Xi += spd_solve(B_xi, U_xi);
    }

    // ====================================================
    // W-STEP: recompute subject-level pi and lam_i
    // ====================================================
    refresh_weights();
    refresh_resid();

    // ====================================================
    // 1-STEP: Inner Newton for Beta (additive IPW)
    // ====================================================
    {
      int inner = 0;
      for (;;) {
        const VectorXd alpha = Eta.head(p_Alpha);
        const VectorXd tau   = Eta.tail(p_Tau);

        VectorXd U1  = VectorXd::Zero(p_Beta);
        MatrixXd Wr1 = MatrixXd::Zero(p_Beta, p_Beta);

        for (int i = 0; i < n - 1; ++i) {
          for (int j = i + 1; j < n; ++j) {
            VectorXd phi(p_Alpha), G_alpha(p_Alpha);
            double h2, sigD2;
            make_phi(i, j, phi, h2, sigD2, G_alpha, alpha);

            for (int k = 0; k < K; ++k) {
              const double ws_i   = ws_val(i, k, tau);
              const double ws_j   = ws_val(j, k, tau);
              const double WS_ijk = ws_i + ws_j;
              const double v_k    = sigD2 + WS_ijk;
              const double V0     = 0.25 * std::max(v_k, tiny);
              const double w      = 1.0 / V0;

              const double lam_yi = (r[k](i) > 0.5) ?
                lam_i[k](i) * y[k](i) : 0.0;
              const double lam_yj = (r[k](j) > 0.5) ?
                lam_i[k](j) * y[k](j) : 0.0;
              const double f1_ipw = 0.5 * (lam_yi + lam_yj);

              const VectorXd x_avg = 0.5 * (x_sub[k].row(i) +
                                            x_sub[k].row(j)).transpose();
              const double h1 = x_avg.dot(Beta);

              U1.noalias()  += x_avg * (w * (f1_ipw - h1));
              Wr1.noalias() += x_avg * x_avg.transpose() * w;
            }
          }
        }

        const VectorXd step     = spd_solve(Wr1, U1);
        const VectorXd Beta_new = Beta + step;
        const double   diff     = (Beta_new - Beta).cwiseAbs().maxCoeff();
        Beta = Beta_new;
        ++inner;
        if (diff < eps || inner > 50) break;
      }
      refresh_resid();
    }

    // ====================================================
    // 2-STEP: Inner Newton for Eta (factorised IPW)
    // ====================================================
    {
      int inner = 0;
      for (;;) {
        const VectorXd alpha = Eta.head(p_Alpha);
        const VectorXd tau   = Eta.tail(p_Tau);

        VectorXd U23  = VectorXd::Zero(p_Eta);
        MatrixXd Wr23 = MatrixXd::Zero(p_Eta, p_Eta);

        for (int i = 0; i < n - 1; ++i) {
          for (int j = i + 1; j < n; ++j) {
            VectorXd phi(p_Alpha), G_alpha(p_Alpha);
            double h2, sigD2;
            make_phi(i, j, phi, h2, sigD2, G_alpha, alpha);

            // --- f^(2): k = 0..K-2, weight = lam_i[k+1](i)*lam_i[k+1](j) ---
            for (int k = 0; k < K - 1; ++k) {
              const double lw2 = lam_i[k + 1](i) * lam_i[k + 1](j);
              if (lw2 < tiny) continue;

              const double d1 = e[k](i)     - e[k](j);
              const double d2 = e[k + 1](i) - e[k + 1](j);
              const double f2 = 0.5 * d1 * d2;

              const double ws_i_k  = ws_val(i, k,     tau);
              const double ws_j_k  = ws_val(j, k,     tau);
              const double ws_i_k1 = ws_val(i, k + 1, tau);
              const double ws_j_k1 = ws_val(j, k + 1, tau);
              const double v_k     = sigD2 + ws_i_k  + ws_j_k;
              const double v_k1    = sigD2 + ws_i_k1 + ws_j_k1;
              const double V0_f2   = 0.25 * (v_k * v_k1 + sigD2 * sigD2);
              const double w2      = lw2 / std::max(V0_f2, tiny);

              // G2: only alpha block non-zero
              VectorXd G2(p_Eta);
              G2.head(p_Alpha)    = G_alpha;
              G2.tail(p_Tau).setZero();

              U23.noalias()  += G2 * (w2 * (f2 - h2));
              Wr23.noalias() += G2 * G2.transpose() * w2;
            }

            // --- f^(3): k = 0..K-1, weight = lam_i[k](i)*lam_i[k](j) ---
            for (int k = 0; k < K; ++k) {
              const double lw3 = lam_i[k](i) * lam_i[k](j);
              if (lw3 < tiny) continue;

              const double d  = e[k](i) - e[k](j);
              const double f3 = 0.5 * d * d;

              const Eigen::Vector4d w_ik  = make_w(i, k);
              const Eigen::Vector4d w_jk  = make_w(j, k);
              const double ws_i = std::exp(tau.dot(w_ik));
              const double ws_j = std::exp(tau.dot(w_jk));
              const double WS   = ws_i + ws_j;
              const double h3   = h2 + 0.5 * WS;  // E[f^(3)] = h2_pair + 0.5*WSpair
              const double v_k  = sigD2 + WS;
              const double V0_f3 = 0.5 * v_k * v_k;
              const double w3   = lw3 / std::max(V0_f3, tiny);

              // G3_tau = w_ik * ws_i + w_jk * ws_j
              VectorXd G3(p_Eta);
              G3.head(p_Alpha) = G_alpha;
              G3.tail(p_Tau)   = w_ik * ws_i + w_jk * ws_j;

              U23.noalias()  += G3 * (w3 * (f3 - h3));
              Wr23.noalias() += G3 * G3.transpose() * w3;
            }
          }
        }

        const VectorXd step    = spd_solve(Wr23, U23);
        const VectorXd Eta_new = Eta + step;
        const double   diff    = (Eta_new - Eta).cwiseAbs().maxCoeff();
        Eta = Eta_new;
        ++inner;
        if (diff < eps || inner > 50) break;
      }
    }

    // ====================================================
    // Convergence check
    // ====================================================
    const double dc1 = (Beta - Beta_old).cwiseAbs().maxCoeff();
    const double dc2 = (Eta  - Eta_old ).cwiseAbs().maxCoeff();
    const double dc3 = (Xi   - Xi_old  ).cwiseAbs().maxCoeff();
    if (std::max({dc1, dc2, dc3}) < eps) converged = true;

  } // end outer while

  // ============================================================
  // SANDWICH VARIANCE
  // ============================================================

  refresh_weights();
  refresh_resid();

  const VectorXd alpha_f = Eta.head(p_Alpha);
  const VectorXd tau_f   = Eta.tail(p_Tau);

  // --- Build averaged Jacobian blocks ---
  MatrixXd Wr1_fin  = MatrixXd::Zero(p_Beta, p_Beta);
  MatrixXd Wr23_fin = MatrixXd::Zero(p_Eta,  p_Eta);

  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      VectorXd phi(p_Alpha), G_alpha(p_Alpha);
      double h2, sigD2;
      make_phi(i, j, phi, h2, sigD2, G_alpha, alpha_f);

      // Beta block
      for (int k = 0; k < K; ++k) {
        const double ws_i   = ws_val(i, k, tau_f);
        const double ws_j   = ws_val(j, k, tau_f);
        const double v_k    = sigD2 + ws_i + ws_j;
        const double V0     = 0.25 * std::max(v_k, tiny);
        const double w      = 1.0 / V0;
        const VectorXd x_avg = 0.5 * (x_sub[k].row(i) +
                                      x_sub[k].row(j)).transpose();
        Wr1_fin.noalias() += x_avg * x_avg.transpose() * w;
      }

      // Eta: f^(2) block
      for (int k = 0; k < K - 1; ++k) {
        const double lw2 = lam_i[k + 1](i) * lam_i[k + 1](j);
        if (lw2 < tiny) continue;
        const double ws_i_k  = ws_val(i, k,     tau_f);
        const double ws_j_k  = ws_val(j, k,     tau_f);
        const double ws_i_k1 = ws_val(i, k + 1, tau_f);
        const double ws_j_k1 = ws_val(j, k + 1, tau_f);
        const double v_k     = sigD2 + ws_i_k  + ws_j_k;
        const double v_k1    = sigD2 + ws_i_k1 + ws_j_k1;
        const double V0_f2   = 0.25 * (v_k * v_k1 + sigD2 * sigD2);
        const double w2      = lw2 / std::max(V0_f2, tiny);
        VectorXd G2(p_Eta);
        G2.head(p_Alpha) = G_alpha;
        G2.tail(p_Tau).setZero();
        Wr23_fin.noalias() += G2 * G2.transpose() * w2;
      }

      // Eta: f^(3) block
      for (int k = 0; k < K; ++k) {
        const double lw3 = lam_i[k](i) * lam_i[k](j);
        if (lw3 < tiny) continue;
        const Eigen::Vector4d w_ik = make_w(i, k);
        const Eigen::Vector4d w_jk = make_w(j, k);
        const double ws_i  = std::exp(tau_f.dot(w_ik));
        const double ws_j  = std::exp(tau_f.dot(w_jk));
        const double WS    = ws_i + ws_j;
        const double v_k   = sigD2 + WS;
        const double V0_f3 = 0.5 * v_k * v_k;
        const double w3    = lw3 / std::max(V0_f3, tiny);
        VectorXd G3(p_Eta);
        G3.head(p_Alpha) = G_alpha;
        G3.tail(p_Tau)   = w_ik * ws_i + w_jk * ws_j;
        Wr23_fin.noalias() += G3 * G3.transpose() * w3;
      }
    }
  }

  // --- Beta individual-level sandwich with diagonal bias correction ---
  MatrixXd M_beta = MatrixXd::Zero(p_Beta, p_Beta);
  MatrixXd D_beta = MatrixXd::Zero(p_Beta, p_Beta);

  for (int i = 0; i < n; ++i) {
    VectorXd c_i = VectorXd::Zero(p_Beta);

    for (int j = 0; j < n; ++j) {
      if (j == i) continue;

      VectorXd phi(p_Alpha), G_alpha(p_Alpha);
      double h2, sigD2;
      make_phi(i, j, phi, h2, sigD2, G_alpha, alpha_f);

      VectorXd a_ij = VectorXd::Zero(p_Beta);
      for (int k = 0; k < K; ++k) {
        if (lam_i[k](i) == 0.0) continue;
        const double ws_i = ws_val(i, k, tau_f);
        const double ws_j = ws_val(j, k, tau_f);
        const double v_k = sigD2 + ws_i + ws_j;
        const double V0 = 0.25 * std::max(v_k, tiny);
        const double h1 = 0.5 * x_sub[k].row(i).dot(Beta)
                        + 0.5 * x_sub[k].row(j).dot(Beta);
        const double resid = lam_i[k](i) * y[k](i) - h1;
        const VectorXd x_avg = 0.5 * (x_sub[k].row(i) +
                                      x_sub[k].row(j)).transpose();
        a_ij.noalias() += x_avg * (0.5 / V0 * resid);
      }
      c_i.noalias() += a_ij;
      D_beta.noalias() += a_ij * a_ij.transpose();
    }

    M_beta.noalias() += c_i * c_i.transpose();
  }

  M_beta.noalias() -= D_beta;

  const MatrixXd Wr1_inv = pinv_sym(Wr1_fin);
  const MatrixXd Sigma_beta = Wr1_inv * M_beta * Wr1_inv;

  // --- Eta Hajek projection sandwich ---
  std::vector<VectorXd> v_eta(n, VectorXd::Zero(p_Eta));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (j == i) continue;

      const int i1 = std::min(i, j);
      const int i2 = std::max(i, j);
      VectorXd W_eta = VectorXd::Zero(p_Eta);

      VectorXd phi(p_Alpha), G_alpha(p_Alpha);
      double h2, sigD2;
      make_phi(i1, i2, phi, h2, sigD2, G_alpha, alpha_f);

      for (int k = 0; k < K - 1; ++k) {
        const double lw2 = lam_i[k + 1](i1) * lam_i[k + 1](i2);
        if (lw2 < tiny) continue;
        const double d1 = e[k](i1) - e[k](i2);
        const double d2 = e[k + 1](i1) - e[k + 1](i2);
        const double f2 = 0.5 * d1 * d2;
        const double ws_i_k = ws_val(i1, k, tau_f);
        const double ws_j_k = ws_val(i2, k, tau_f);
        const double ws_i_k1 = ws_val(i1, k + 1, tau_f);
        const double ws_j_k1 = ws_val(i2, k + 1, tau_f);
        const double v_k = sigD2 + ws_i_k + ws_j_k;
        const double v_k1 = sigD2 + ws_i_k1 + ws_j_k1;
        const double V0_f2 = 0.25 * (v_k * v_k1 + sigD2 * sigD2);
        const double w2 = lw2 / std::max(V0_f2, tiny);
        VectorXd G2(p_Eta);
        G2.head(p_Alpha) = G_alpha;
        G2.tail(p_Tau).setZero();
        W_eta.noalias() += G2 * (w2 * (f2 - h2));
      }

      for (int k = 0; k < K; ++k) {
        const double lw3 = lam_i[k](i1) * lam_i[k](i2);
        if (lw3 < tiny) continue;
        const double d = e[k](i1) - e[k](i2);
        const double f3 = 0.5 * d * d;
        const Eigen::Vector4d w_ik = make_w(i1, k);
        const Eigen::Vector4d w_jk = make_w(i2, k);
        const double ws_i = std::exp(tau_f.dot(w_ik));
        const double ws_j = std::exp(tau_f.dot(w_jk));
        const double WS = ws_i + ws_j;
        const double h3 = h2 + 0.5 * WS;
        const double v_k = sigD2 + WS;
        const double V0_f3 = 0.5 * v_k * v_k;
        const double w3 = lw3 / std::max(V0_f3, tiny);
        VectorXd G3(p_Eta);
        G3.head(p_Alpha) = G_alpha;
        G3.tail(p_Tau) = w_ik * ws_i + w_jk * ws_j;
        W_eta.noalias() += G3 * (w3 * (f3 - h3));
      }

      v_eta[i].noalias() += W_eta / static_cast<double>(n - 1);
    }
  }

  VectorXd veta_bar = VectorXd::Zero(p_Eta);
  for (int i = 0; i < n; ++i) veta_bar += v_eta[i];
  veta_bar /= static_cast<double>(n);

  MatrixXd Sigma_W_eta = MatrixXd::Zero(p_Eta, p_Eta);
  for (int i = 0; i < n; ++i) {
    const VectorXd dv = v_eta[i] - veta_bar;
    Sigma_W_eta.noalias() += dv * dv.transpose();
  }
  Sigma_W_eta *= 4.0 / static_cast<double>(n - 1);

  const MatrixXd Wr23_inv = pinv_sym(Wr23_fin / static_cast<double>(n_pairs));
  const MatrixXd Sigma_eta = Wr23_inv * Sigma_W_eta * Wr23_inv
                             / static_cast<double>(n);

  MatrixXd Sigma_theta = MatrixXd::Zero(p_BE, p_BE);
  Sigma_theta.block(0, 0, p_Beta, p_Beta) = Sigma_beta;
  Sigma_theta.block(p_Beta, p_Beta, p_Eta, p_Eta) = Sigma_eta;
  const VectorXd sigma_theta = Sigma_theta.diagonal();

  // --- Subject-level cluster-robust sandwich for Xi ---
  MatrixXd Bxi_fin = MatrixXd::Zero(p_Xi, p_Xi);
  MatrixXd Sxi_fin = MatrixXd::Zero(p_Xi, p_Xi);

  for (int i = 0; i < n; ++i) {
    VectorXd s_i = VectorXd::Zero(p_Xi);
    for (int t = 2; t <= K; ++t) {
      const int off = 2 * (t - 2);
      if (r[t - 2](i) != 1.0) continue;
      const double r_cur  = r[t - 1](i);
      const double y_prev = y[t - 2](i);
      Eigen::Vector2d g_t; g_t(0) = 1.0; g_t(1) = y_prev;
      const Eigen::Vector2d xi_t = Xi.segment<2>(off);
      const double p_t = 1.0 / (1.0 + std::exp(-g_t.dot(xi_t)));
      Bxi_fin.block<2, 2>(off, off).noalias() +=
        g_t * g_t.transpose() * p_t * (1.0 - p_t);
      s_i.segment(off, 2).noalias() += g_t * (r_cur - p_t);
    }
    Sxi_fin.noalias() += s_i * s_i.transpose();
  }

  const double dn     = static_cast<double>(n);
  const MatrixXd Bxi_inv = pinv_sym(Bxi_fin / dn);
  const MatrixXd Sigma_xi_logit = Bxi_inv * (Sxi_fin / dn) * Bxi_inv / dn;
  const VectorXd sigma_xi_logit = Sigma_xi_logit.diagonal();

  // --------------------------------------------------------
  // 6. Pack and return
  // --------------------------------------------------------
  NumericVector theta_out(p);
  for (int r_ = 0; r_ < p_Beta; ++r_) theta_out[r_]          = Beta(r_);
  for (int r_ = 0; r_ < p_Eta;  ++r_) theta_out[p_Beta + r_] = Eta(r_);
  for (int r_ = 0; r_ < p_Xi;   ++r_) theta_out[p_BE   + r_] = Xi(r_);

  return Rcpp::List::create(
    _["theta.est"]         = theta_out,
    _["Beta"]              = Rcpp::wrap(Beta),
    _["Eta"]               = Rcpp::wrap(Eta),
    _["Xi"]                = Rcpp::wrap(Xi),
    _["Sigma.theta"]       = Rcpp::wrap(Sigma_theta),
    _["sigma.theta"]       = Rcpp::wrap(sigma_theta),
    _["Sigma.xi.logit"]    = Rcpp::wrap(Sigma_xi_logit),
    _["sigma.xi.logit"]    = Rcpp::wrap(sigma_xi_logit),
    _["iter"]              = iter,
    _["converged"]         = converged
  );
}
