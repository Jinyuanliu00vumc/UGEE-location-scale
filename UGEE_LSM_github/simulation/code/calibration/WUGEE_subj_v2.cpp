// WUGEE_subj.cpp
// ============================================================
// Weighted UGEE — Subject-level Missing Model
//
// Companion to WUGEE_v3.cpp (pairwise model).  Here dropout is
// modelled at the individual subject level:
//
//   logit(p_{it}(xi_t)) = xi_{0t} + xi_{1t} * y_{i,t-1}
//   pi_{ik}(xi) = prod_{t=2}^{k} p_{it}(xi_t)   [marginal survival]
//
// KEY DIFFERENCE from the pairwise model
// -----------------------------------------------
// f^(1) location functional uses the ADDITIVE IPW form (tex eq. s1):
//
//   f^(1,IPW)_{ij,k} = 0.5*(r_{ik}*y_{ik}/pi_{ik} + r_{jk}*y_{jk}/pi_{jk})
//
// This is unbiased for x_k^T Beta for ALL pairs at ALL times, even
// when ONE subject is missing.  The corresponding beta EE is therefore
// summed over ALL n*(n-1)/2 pairs x ALL K times (not just jointly-observed
// ones), capturing information from partially-observed pairs.
//
// f^(2) and f^(3) (variance components) still require JOINT observation
// because they involve cross-subject differences/products.  Their IPW
// weights use the FACTORISED subject-level probabilities:
//
//   lambda^(2)_{ij,k} = r_{i,k+1}*r_{j,k+1} / (pi_{i,k+1} * pi_{j,k+1})
//   lambda^(3)_{ij,k} = r_{ik}*r_{jk}        / (pi_{ik}    * pi_{jk}   )
//
// PARAMETER BLOCKS
// -----------------------------------------------
//   Beta = (beta0, beta1, beta2)               [p_Beta  = 3]
//   Eta  = (alpha0, alpha1, tau0, tau1, tau2)  [p_Eta   = 5]
//   Xi   = (xi_{0,2}, xi_{1,2}, ...,
//            xi_{0,K}, xi_{1,K})               [p_Xi    = 2*(K-1)]
//
// ITERATIVE ALGORITHM
// -----------------------------------------------
//   Outer loop until convergence:
//     M-step : Newton update for Xi (subject logistic score)
//     W-step : Recompute pi_{ik} and individual IPW weights
//     1-step : Inner Newton loop for Beta (additive f^(1,IPW) EE)
//     2-step : Inner Newton loop for Eta  (factorised f^(2)+f^(3) EE)
//   Convergence: max|param_new - param_old| < eps
//
// SANDWICH VARIANCE
// -----------------------------------------------
//   Beta/Eta: joint Hajek projection sandwich (mirrors UGEE_v2.cpp exactly)
//     v_i  = (1/(n-1)) * sum_{j != i} psi_{ij}(theta_hat)   [joint, length p_Beta+p_Eta]
//     Sigma_U     = 4/(n-1) * sum_i v_i v_i^T               [no centering]
//     B_avg       = block_diag(Wr1_fin, Wr23_fin) / n_pairs
//     Sigma_theta = B_avg^{-1} Sigma_U B_avg^{-1} / n
//
//   Xi: separate subject-level cluster-robust sandwich
//     B_xi = (1/n) sum_i sum_t g_{it} g_{it}^T p_{it}(1-p_{it})
//     S_xi = (1/n) sum_i s_i s_i^T   [s_i = sum_t g_{it}(r_{it}-p_{it})]
//     Sigma_xi = B_xi^{-1} S_xi B_xi^{-1} / n
//
// INPUTS
// -----------------------------------------------
//   y_ki_all   : list of K NumericVectors length n
//   r_indiv_all: list of K NumericVectors length n  (r_{ik} in {0,1})
//   ui_R       : length-n binary covariate u_i
//   t_star     : change-point
//   Beta_init, Eta_init, Xi_init : starting values
//   eps, max_iter
//
// OUTPUTS (List)
//   theta.est          : (Beta, Eta, Xi) concatenated
//   Beta, Eta, Xi      : separate blocks
//   Sigma.theta        : Hajek sandwich for (Beta, Eta) [(p_Beta+p_Eta)^2]
//   sigma.theta        : diagonal of Sigma.theta
//   Sigma.xi.logit     : subject cluster-robust sandwich for Xi
//   sigma.xi.logit     : diagonal of Sigma.xi.logit
//   iter, converged
// ============================================================

// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;

// -------------------------------------------------------
// Helper: safe Cholesky solve with LDLT fallback
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
Rcpp::List wugee_subj(
    Rcpp::List          y_ki_all,
    Rcpp::List          r_indiv_all,   // K-list of length-n vectors: r_{ik}
    Rcpp::NumericVector ui_R,
    double              t_star,
    Rcpp::NumericVector Beta_init,
    Rcpp::NumericVector Eta_init,
    Rcpp::NumericVector Xi_init,
    double eps      = 1e-5,
    int    max_iter = 500
) {
  // --------------------------------------------------------
  // 0. Dimensions & validation
  // --------------------------------------------------------
  const int K = y_ki_all.size();
  if (K < 2) stop("K must be >= 2.");
  if (r_indiv_all.size() != K) stop("r_indiv_all must have length K.");
  
  Rcpp::NumericVector y0 = y_ki_all[0];
  const int n = y0.size();
  if (n < 2) stop("n must be >= 2.");
  
  const int p_Beta = 3;
  const int p_Eta  = 5;
  const int p_Xi   = 2 * (K - 1);
  const int p_BE   = p_Beta + p_Eta;         // joint sandwich dim
  const int p      = p_BE + p_Xi;
  
  if (Beta_init.size() != p_Beta) stop("Beta_init must have length 3.");
  if (Eta_init.size()  != p_Eta)  stop("Eta_init must have length 5.");
  if (Xi_init.size()   != p_Xi)   stop("Xi_init must have length 2*(K-1).");
  
  // --------------------------------------------------------
  // 1. Copy inputs into Eigen
  // --------------------------------------------------------
  std::vector<VectorXd> y(K, VectorXd(n));
  for (int k = 0; k < K; ++k) {
    Rcpp::NumericVector yk = y_ki_all[k];
    if (yk.size() != n) stop("All y_ki_all[[k]] must have length n.");
    for (int i = 0; i < n; ++i) y[k](i) = yk[i];
  }
  
  // r[k](i) = r_{i,k+1} (0-indexed k corresponds to time t=k+1)
  std::vector<VectorXd> r(K, VectorXd(n));
  for (int k = 0; k < K; ++k) {
    Rcpp::NumericVector rk = r_indiv_all[k];
    if (rk.size() != n) stop("All r_indiv_all[[k]] must have length n.");
    for (int i = 0; i < n; ++i) r[k](i) = rk[i];
  }
  
  VectorXd ui(n);
  for (int i = 0; i < n; ++i) ui(i) = ui_R[i];
  
  const int n_pairs = n * (n - 1) / 2;
  
  // --------------------------------------------------------
  // 2. Time design matrix (same as WUGEE_v3)
  // --------------------------------------------------------
  MatrixXd k_vecs(K, 3);
  for (int k = 0; k < K; ++k) {
    const double kk  = static_cast<double>(k + 1);
    const double kpl = std::max(0.0, kk - t_star);
    k_vecs.row(k) << 1.0, kk, kpl;
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
  
  // pi_mat[k](i) = pi_{i,k+1}: marginal prob of observing subject i at time k+1
  std::vector<VectorXd> pi_mat(K, VectorXd::Ones(n));
  // lam_i[k](i)  = r_{i,k+1} / pi_{i,k+1}   (individual IPW weight)
  std::vector<VectorXd> lam_i(K, VectorXd::Zero(n));
  
  std::vector<VectorXd> e(K, VectorXd(n));   // residuals
  
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
    // M-STEP: Newton update for Xi (subject-level logistic)
    //
    // For each t = 2..K, for each subject i with r_{i,t-1} = 1:
    //   g_{it} = (1, y_{i,t-1})^T
    //   p_{it} = expit(xi_{0t} + xi_{1t} * y_{i,t-1})
    //   score  = g_{it} * (r_{it} - p_{it})
    //   hessian= g_{it} g_{it}^T p_{it}(1 - p_{it})
    // ====================================================
    {
      VectorXd U_xi = VectorXd::Zero(p_Xi);
      MatrixXd B_xi = MatrixXd::Zero(p_Xi, p_Xi);
      
      for (int t = 2; t <= K; ++t) {
        const int off = 2 * (t - 2);
        Eigen::Vector2d U_t = Eigen::Vector2d::Zero();
        Eigen::Matrix2d B_t = Eigen::Matrix2d::Zero();
        
        for (int i = 0; i < n; ++i) {
          // Subject must have been observed at t-1 (0-indexed: k = t-2)
          if (r[t - 2](i) != 1.0) continue;
          
          const double r_cur   = r[t - 1](i);
          const double y_prev  = y[t - 2](i);
          Eigen::Vector2d g_t;
          g_t(0) = 1.0;
          g_t(1) = y_prev;
          
          const Eigen::Vector2d xi_t = Xi.segment<2>(off);
          const double eta_t = g_t.dot(xi_t);
          const double p_t   = 1.0 / (1.0 + std::exp(-eta_t));
          
          U_t.noalias() += g_t * (r_cur - p_t);
          B_t.noalias() += g_t * g_t.transpose() * p_t * (1.0 - p_t);
        }
        
        U_xi.segment<2>(off) = U_t;
        B_xi.block<2, 2>(off, off) = B_t;
      }
      
      // Small ridge for stability
      for (int d = 0; d < p_Xi; ++d) B_xi(d, d) += 1e-8;
      Xi += spd_solve(B_xi, U_xi);
    }
    
    // ====================================================
    // W-STEP: recompute subject-level marginal weights pi_{ik}
    //
    // pi_{i,1} = 1 (all observed at t=1)
    // pi_{i,k} = pi_{i,k-1} * p_{i,k}  for k=2..K
    // lam_i[k](i) = r[k](i) / pi_{ik}
    // ====================================================
    {
      for (int k = 0; k < K; ++k) {
        pi_mat[k].setOnes();
        lam_i[k].setZero();
      }
      // t=1 (k=0): pi=1, lam = r[0](i) (all 1 since fully observed)
      lam_i[0] = r[0].cwiseMax(0.0);  // should all be 1
      
      for (int i = 0; i < n; ++i) {
        double pi_cur = 1.0;
        for (int t = 2; t <= K; ++t) {
          const int k_idx = t - 1;   // 0-indexed
          // Update pi regardless of dropout (marginal probability)
          const double y_prev = y[t - 2](i);
          Eigen::Vector2d g_t; g_t(0) = 1.0; g_t(1) = y_prev;
          const int off = 2 * (t - 2);
          const Eigen::Vector2d xi_t = Xi.segment<2>(off);
          // Only multiply by p_{it} if subject was at risk at t-1.
          // Once dropped (monotone), pi_mat and lam_i are both set to 0
          // so that: (i) pi_mat[k](i) is never a stale nonzero for a
          // dropped subject, and (ii) the exact-zero guards in the Eta
          // step correctly skip their residual contributions.
          if (r[t - 2](i) == 1.0) {
            const double p_t = 1.0 / (1.0 + std::exp(-g_t.dot(xi_t)));
            pi_cur *= p_t;
            pi_mat[k_idx](i) = pi_cur;
          } else {
            // Subject already dropped: marginal probability is 0
            pi_cur = 0.0;
            pi_mat[k_idx](i) = 0.0;
          }
          // lam_i = r_{it} / pi_{it}
          lam_i[k_idx](i) = (pi_cur > tiny) ? r[k_idx](i) / pi_cur : 0.0;
        }
      }
    }
    
    // ====================================================
    // Precompute residuals e[k](i) = y_{ik} - x_k^T Beta
    //
    // NOTE: residuals are computed for ALL subjects, including those not
    // observed at time k.  Unobserved subjects carry latent y[k](i)
    // values from the DGP buffer.  Their contributions to f^(2) and f^(3)
    // are zeroed out in the Eta step via the exact-zero IPW weight guards
    // (lw2 == 0.0 / lw3 == 0.0 checks below), which are reliable because
    // lam_i[k](i) is explicitly set to 0.0 in the W-step whenever
    // r[k](i) == 0.  Do not alter the lam_i initialisation without
    // revisiting these guards.
    // ====================================================
    for (int k = 0; k < K; ++k) {
      const double h1 = k_vecs.row(k).dot(Beta);
      for (int i = 0; i < n; ++i) e[k](i) = y[k](i) - h1;
    }
    
    // ====================================================
    // 1-STEP: Inner Newton loop for Beta
    //
    // Working variance V0 = Var(f^(1)_{ij,k}) under complete data.
    // Must use 1/V0 (not 1/V0_ipw) to keep E[score]=0 under MAR.
    // ====================================================
    bool inner_conv_b = false;
    {
      VectorXd sigb(n);
      for (int i = 0; i < n; ++i)
        sigb(i) = std::exp(Eta(0) + Eta(1) * ui(i));
      
      int  inner        = 0;
      for (;;) {
        VectorXd U1  = VectorXd::Zero(p_Beta);
        MatrixXd Wr1 = MatrixXd::Zero(p_Beta, p_Beta);
        
        for (int i = 0; i < n - 1; ++i) {
          for (int j = i + 1; j < n; ++j) {
            const double h2ij = 0.5 * (sigb(i) + sigb(j));
            for (int k = 0; k < K; ++k) {
              const double WS_k   = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
              const double V0     = 0.5 * h2ij + 0.5 * WS_k;
              const double f1_ipw = 0.5 * (lam_i[k](i) * y[k](i)
                + lam_i[k](j) * y[k](j));
              const double h1     = k_vecs.row(k).dot(Beta);
              const double w      = 1.0 / std::max(V0, tiny);
              for (int r_ = 0; r_ < p_Beta; ++r_)
                U1(r_) += k_vecs(k, r_) * w * (f1_ipw - h1);
              for (int r_ = 0; r_ < p_Beta; ++r_)
                for (int c = 0; c < p_Beta; ++c)
                  Wr1(r_, c) += k_vecs(k, r_) * w * k_vecs(k, c);
            }
          }
        }
        
        const VectorXd step     = spd_solve(Wr1, U1);
        const VectorXd Beta_new = Beta + step;
        const double   diff     = (Beta_new - Beta).cwiseAbs().maxCoeff();
        Beta = Beta_new;
        ++inner;
        if (diff < eps) { inner_conv_b = true; break; }
        if (inner >= 200) {
          Rcpp::warning("Beta inner loop hit cap (200) at outer iter %d; "
                          "outer convergence suppressed.", iter);
          break;
        }
      }
      // inner_conv_b feeds into the final convergence check at end of outer iter
      
      for (int k = 0; k < K; ++k) {
        const double h1 = k_vecs.row(k).dot(Beta);
        for (int i = 0; i < n; ++i) e[k](i) = y[k](i) - h1;
      }
    }
    
    // ====================================================
    // 2-STEP: Inner Newton loop for Eta
    //
    // f^(2) and f^(3) require JOINT observation of both i,j.
    // Factorised subject-level IPW weights:
    //   lambda^(2)_{ij,k} = r_{i,k+1}*r_{j,k+1} / (pi_{i,k+1}*pi_{j,k+1})
    //   lambda^(3)_{ij,k} = r_{ik}*r_{jk}        / (pi_{ik}*pi_{jk})
    //
    // Note: these are exactly lam_i[k+1](i)*lam_i[k+1](j)  and
    //                          lam_i[k](i)*lam_i[k](j)
    //       but only nonzero when both subjects are observed.
    // ====================================================
    bool inner_conv_e = false;
    {
      int  inner        = 0;
      for (;;) {
        const double a0 = Eta(0), a1 = Eta(1);
        VectorXd sigb(n), ui_sigb(n);
        for (int i = 0; i < n; ++i) {
          sigb(i)    = std::exp(a0 + a1 * ui(i));
          ui_sigb(i) = ui(i) * sigb(i);
        }
        
        VectorXd U23  = VectorXd::Zero(p_Eta);
        MatrixXd Wr23 = MatrixXd::Zero(p_Eta, p_Eta);
        
        for (int i = 0; i < n - 1; ++i) {
          for (int j = i + 1; j < n; ++j) {
            const double h2ij   = 0.5 * (sigb(i)    + sigb(j));
            const double dh2_a0 = h2ij;
            const double dh2_a1 = 0.5 * (ui_sigb(i) + ui_sigb(j));
            const double sigD2  = 2.0 * h2ij;
            
            // --- f^(2): k = 0..K-2, weight = lam_i[k+1](i)*lam_i[k+1](j) ---
            for (int k = 0; k < K - 1; ++k) {
              // Factorised weight for f^(2): both must be observed at k+1
              const double lw2 = lam_i[k + 1](i) * lam_i[k + 1](j);
              if (lw2 == 0.0) continue;
              
              const double d1    = e[k](i)     - e[k](j);
              const double d2    = e[k + 1](i) - e[k + 1](j);
              const double f2    = 0.5 * d1 * d2;
              const double WS_k  = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
              const double WS_k1 = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k + 1)));
              const double v_k   = sigD2 + 2.0 * WS_k;
              const double v_k1  = sigD2 + 2.0 * WS_k1;
              const double V0_f2 = 0.25 * (v_k * v_k1 + sigD2 * sigD2);
              const double w2    = lw2 / std::max(V0_f2, tiny);
              
              U23(0)    += dh2_a0 * w2 * (f2 - h2ij);
              U23(1)    += dh2_a1 * w2 * (f2 - h2ij);
              Wr23(0,0) += dh2_a0 * dh2_a0 * w2;
              Wr23(0,1) += dh2_a0 * dh2_a1 * w2;
              Wr23(1,0) += dh2_a1 * dh2_a0 * w2;
              Wr23(1,1) += dh2_a1 * dh2_a1 * w2;
            }
            
            // --- f^(3): k = 0..K-1, weight = lam_i[k](i)*lam_i[k](j) ---
            for (int k = 0; k < K; ++k) {
              const double lw3 = lam_i[k](i) * lam_i[k](j);
              if (lw3 == 0.0) continue;
              
              const double d    = e[k](i) - e[k](j);
              const double f3   = 0.5 * d * d;
              const double WS_k = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
              const double h3   = h2ij + WS_k;
              const double v_k  = sigD2 + 2.0 * WS_k;
              const double V0_f3 = 0.5 * v_k * v_k;
              const double w3    = lw3 / std::max(V0_f3, tiny);
              
              VectorXd G3(p_Eta);
              G3(0) = dh2_a0;
              G3(1) = dh2_a1;
              G3(2) = k_vecs(k, 0) * WS_k;
              G3(3) = k_vecs(k, 1) * WS_k;
              G3(4) = k_vecs(k, 2) * WS_k;
              
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
        if (diff < eps) { inner_conv_e = true; break; }
        if (inner >= 200) {
          Rcpp::warning("Eta inner loop hit cap (200) at outer iter %d; "
                          "outer convergence suppressed.", iter);
          break;
        }
      }
    }
    // inner_conv_e feeds into the final convergence check at end of outer iter
    
    // ====================================================
    // Convergence check
    // Declare converged only when ALL of:
    //   (a) Beta inner loop converged (inner_conv_b), AND
    //   (b) Eta  inner loop converged (inner_conv_e), AND
    //   (c) all outer parameter blocks moved by less than eps.
    // ====================================================
    {
      const double d1 = (Beta - Beta_old).cwiseAbs().maxCoeff();
      const double d2 = (Eta  - Eta_old ).cwiseAbs().maxCoeff();
      const double d3 = (Xi   - Xi_old  ).cwiseAbs().maxCoeff();
      converged = inner_conv_b && inner_conv_e &&
        (std::max({d1, d2, d3}) < eps);
    }
    
  } // end outer while
  
  // ============================================================
  // SANDWICH VARIANCE
  //
  // Both Beta and Eta use the Hajek projection sandwich for
  // weighted incomplete U-statistics.
  //
  // For both blocks:
  //   psi_{ij} = pair-level score kernel (symmetric in i,j)
  //   B        = (1/n_pairs) * sum_{i<j} E[-d psi_{ij}/d theta]   [Jacobian]
  //   v_i      = (1/(n-1)) * sum_{j!=i} psi_{ij}                  [Hajek projection]
  //   Sigma_W  = 4/(n-1) * sum_i (v_i - v_bar)(v_i - v_bar)^T
  //   Sigma    = B^{-1} * Sigma_W * B^{-1} / n
  //
  // Xi: separate subject-level cluster-robust sandwich.
  // ============================================================
  
  // --- Precompute sigb_f and residuals ---
  VectorXd sigb_f(n), ui_sigb_f(n);
  for (int i = 0; i < n; ++i) {
    sigb_f(i)    = std::exp(Eta(0) + Eta(1) * ui(i));
    ui_sigb_f(i) = ui(i) * sigb_f(i);
  }
  for (int k = 0; k < K; ++k) {
    const double h1 = k_vecs.row(k).dot(Beta);
    for (int i = 0; i < n; ++i) e[k](i) = y[k](i) - h1;
  }
  
  // --- Recompute lam_i at final Xi ---
  {
    for (int k = 0; k < K; ++k) { pi_mat[k].setOnes(); lam_i[k].setZero(); }
    lam_i[0] = r[0].cwiseMax(0.0);
    for (int i = 0; i < n; ++i) {
      double pi_cur = 1.0;
      for (int t = 2; t <= K; ++t) {
        const int k_idx = t - 1;
        if (r[t-2](i) == 1.0) {
          Eigen::Vector2d g_t; g_t(0) = 1.0; g_t(1) = y[t-2](i);
          const Eigen::Vector2d xi_t = Xi.segment<2>(2*(t-2));
          pi_cur *= 1.0 / (1.0 + std::exp(-g_t.dot(xi_t)));
          pi_mat[k_idx](i) = pi_cur;
        } else {
          // Subject already dropped: set to 0 (not stale pi_cur)
          pi_cur = 0.0;
          pi_mat[k_idx](i) = 0.0;
        }
        lam_i[k_idx](i)  = (pi_cur > tiny) ? r[k_idx](i) / pi_cur : 0.0;
      }
    }
  }
  
  // ============================================================
  // SANDWICH VARIANCE
  //
  // BETA: individual-level sandwich.
  //   The additive IPW score decomposes as U_beta = sum_i c_i where
  //     c_i = sum_{j!=i, k}  x_k / (2*V_{ij,k}) * (lam_i[k](i)*y[k](i) - h1_k)
  //   Only subject i's OWN IPW outcome enters c_i; this is mean-zero
  //   under MAR since E[lam_ik * y_ik] = h1_k.  Using the full pair
  //   kernel f1_ipw = 0.5*(lam_ik*y_ik + lam_jk*y_jk) in the Hajek
  //   projection inflates Sigma_U because the j-side term is NOT
  //   mean-zero when j is unobserved (lam_jk=0 gives expectation -h1/2).
  //
  //   Jacobian: Wr1_fin = sum_{i<j,k} x_k x_k^T / V_{ij,k}  [raw sum]
  //   Meat:     M_beta  = (1/n) * sum_i c_i c_i^T
  //   Sandwich: Wr1_fin^{-1} * M_beta * Wr1_fin^{-1}
  //
  // ETA: Hajek projection sandwich (genuine order-2 U-statistic).
  //   Eta pair kernels use pairwise IPW lam_ik*lam_jk which is zero
  //   whenever either subject is unobserved — so those pairs are
  //   correctly skipped and the kernel is mean-zero.
  //
  //   B_eta  = Wr23_fin / n_pairs
  //   v_eta_i = (1/(n-1)) * sum_{j!=i} psi_eta_{ij}
  //   Sigma_W_eta = 4/(n-1) * sum_i (v_eta_i - vbar)(v_eta_i - vbar)^T
  //   Sigma_eta = B_eta^{-1} * Sigma_W_eta * B_eta^{-1} / n
  //
  // Xi: separate subject-level cluster-robust sandwich (unchanged).
  // ============================================================
  
  // --- Jacobians (shared loop) ---
  MatrixXd Wr1_fin  = MatrixXd::Zero(p_Beta, p_Beta);
  MatrixXd Wr23_fin = MatrixXd::Zero(p_Eta,  p_Eta);
  
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double h2ij   = 0.5 * (sigb_f(i) + sigb_f(j));
      const double dh2_a0 = h2ij;
      const double dh2_a1 = 0.5 * (ui_sigb_f(i) + ui_sigb_f(j));
      const double sigD2  = 2.0 * h2ij;
      
      for (int k = 0; k < K; ++k) {
        const double WS_k = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        const double V0   = 0.5 * h2ij + 0.5 * WS_k;
        const double w    = 1.0 / std::max(V0, tiny);
        for (int r_ = 0; r_ < p_Beta; ++r_)
          for (int c = 0; c < p_Beta; ++c)
            Wr1_fin(r_, c) += k_vecs(k, r_) * w * k_vecs(k, c);
      }
      for (int k = 0; k < K - 1; ++k) {
        const double lw2 = lam_i[k+1](i) * lam_i[k+1](j);
        if (lw2 == 0.0) continue;
        const double WS_k  = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        const double WS_k1 = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k+1)));
        const double v_k   = sigD2+2.0*WS_k, v_k1 = sigD2+2.0*WS_k1;
        const double V0_f2 = 0.25*(v_k*v_k1 + sigD2*sigD2);
        const double w2    = lw2 / std::max(V0_f2, tiny);
        Wr23_fin(0,0)+=dh2_a0*dh2_a0*w2; Wr23_fin(0,1)+=dh2_a0*dh2_a1*w2;
        Wr23_fin(1,0)+=dh2_a1*dh2_a0*w2; Wr23_fin(1,1)+=dh2_a1*dh2_a1*w2;
      }
      for (int k = 0; k < K; ++k) {
        const double lw3 = lam_i[k](i) * lam_i[k](j);
        if (lw3 == 0.0) continue;
        const double WS_k  = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        const double v_k   = sigD2+2.0*WS_k;
        const double V0_f3 = 0.5*v_k*v_k;
        const double w3    = lw3 / std::max(V0_f3, tiny);
        VectorXd G3(p_Eta);
        G3(0)=dh2_a0; G3(1)=dh2_a1;
        G3(2)=k_vecs(k,0)*WS_k; G3(3)=k_vecs(k,1)*WS_k; G3(4)=k_vecs(k,2)*WS_k;
        Wr23_fin.noalias() += G3 * G3.transpose() * w3;
      }
    }
  }
  
  // ---- BETA: individual-level sandwich with bias correction ----
  //
  // U_beta = sum_i c_i  where  c_i = sum_{j!=i, k:r_ik=1} A_{ij,k}
  //   A_{ij,k} = x_k / (2*V_{ij,k}) * (lam_ik*y_ik - h1_k)
  //
  // M_beta = sum_i c_i c_i^T  estimates Var(U_beta) but has O(1/n) bias:
  //   E[c_i^2] = (n-1)^2 * Var(h_i) + (n-1) * E[A_{ij}^2]
  // The second term inflates M_beta.  Subtract it:
  //   M_beta_corrected = sum_i c_i c_i^T - sum_i sum_{j!=i} a_ij a_ij^T
  // where a_ij = sum_k A_{ij,k}  (the per-pair contribution to c_i).
  // This leaves only the (n-1)^2 * Var(h_i) term which correctly estimates
  // Var(U_beta) = Var(sum_i c_i).
  MatrixXd M_beta  = MatrixXd::Zero(p_Beta, p_Beta);
  MatrixXd D_beta  = MatrixXd::Zero(p_Beta, p_Beta);  // diagonal correction
  for (int i = 0; i < n; ++i) {
    VectorXd c_i = VectorXd::Zero(p_Beta);
    for (int j = 0; j < n; ++j) {
      if (j == i) continue;
      const double h2ij = 0.5 * (sigb_f(i) + sigb_f(j));
      VectorXd a_ij = VectorXd::Zero(p_Beta);
      for (int k = 0; k < K; ++k) {
        if (lam_i[k](i) == 0.0) continue;
        const double WS_k  = std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        const double V0    = 0.5 * h2ij + 0.5 * WS_k;
        const double h1    = k_vecs.row(k).dot(Beta);
        const double w     = 0.5 / std::max(V0, tiny);
        const double resid = lam_i[k](i) * y[k](i) - h1;
        for (int r_ = 0; r_ < p_Beta; ++r_)
          a_ij(r_) += k_vecs(k, r_) * w * resid;
      }
      c_i.noalias()   += a_ij;
      D_beta.noalias() += a_ij * a_ij.transpose();  // collect diagonal bias
    }
    M_beta.noalias() += c_i * c_i.transpose();
  }
  // Bias-corrected meat: remove the O(n^2) diagonal inflation
  M_beta.noalias() -= D_beta;
  // No division by n here: Var(hat_beta) = Wr1^{-1} * sum_i(c_i c_i^T) * Wr1^{-1}
  // because Var(U1) = Var(sum_i c_i) = sum_i Var(c_i) estimated by sum_i c_i c_i^T.
  
  const MatrixXd Wr1_inv    = pinv_sym(Wr1_fin);
  const MatrixXd Sigma_beta = Wr1_inv * M_beta * Wr1_inv;
  
  // ---- ETA: Hajek projection sandwich ----
  std::vector<VectorXd> v_eta(n, VectorXd::Zero(p_Eta));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (j == i) continue;
      const int i1 = std::min(i,j), i2 = std::max(i,j);
      const double h2ij   = 0.5*(sigb_f(i1)+sigb_f(i2));
      const double dh2_a0 = h2ij;
      const double dh2_a1 = 0.5*(ui_sigb_f(i1)+ui_sigb_f(i2));
      const double sigD2  = 2.0*h2ij;
      VectorXd W_eta = VectorXd::Zero(p_Eta);
      for (int k = 0; k < K-1; ++k) {
        const double lw2 = lam_i[k+1](i1)*lam_i[k+1](i2);
        if (lw2==0.0) continue;
        const double d1=e[k](i1)-e[k](i2), d2=e[k+1](i1)-e[k+1](i2);
        const double f2=0.5*d1*d2;
        const double WS_k =std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        const double WS_k1=std::exp(Eta.segment<3>(2).dot(k_vecs.row(k+1)));
        const double v_k=sigD2+2.0*WS_k, v_k1=sigD2+2.0*WS_k1;
        const double V0_f2=0.25*(v_k*v_k1+sigD2*sigD2);
        const double w2=lw2/std::max(V0_f2,tiny);
        W_eta(0)+=dh2_a0*w2*(f2-h2ij); W_eta(1)+=dh2_a1*w2*(f2-h2ij);
      }
      for (int k = 0; k < K; ++k) {
        const double lw3=lam_i[k](i1)*lam_i[k](i2);
        if (lw3==0.0) continue;
        const double d=e[k](i1)-e[k](i2), f3=0.5*d*d;
        const double WS_k=std::exp(Eta.segment<3>(2).dot(k_vecs.row(k)));
        const double h3=h2ij+WS_k, v_k=sigD2+2.0*WS_k;
        const double V0_f3=0.5*v_k*v_k, w3=lw3/std::max(V0_f3,tiny);
        VectorXd G3(p_Eta);
        G3(0)=dh2_a0; G3(1)=dh2_a1;
        G3(2)=k_vecs(k,0)*WS_k; G3(3)=k_vecs(k,1)*WS_k; G3(4)=k_vecs(k,2)*WS_k;
        W_eta.noalias() += G3*(w3*(f3-h3));
      }
      v_eta[i].noalias() += W_eta / static_cast<double>(n-1);
    }
  }
  
  VectorXd veta_bar = VectorXd::Zero(p_Eta);
  for (int i = 0; i < n; ++i) veta_bar += v_eta[i];
  veta_bar /= static_cast<double>(n);
  MatrixXd Sigma_W_eta = MatrixXd::Zero(p_Eta, p_Eta);
  for (int i = 0; i < n; ++i) {
    const VectorXd dv = v_eta[i] - veta_bar;
    Sigma_W_eta.noalias() += dv*dv.transpose();
  }
  Sigma_W_eta *= 4.0 / static_cast<double>(n-1);
  
  const MatrixXd Wr23_inv   = pinv_sym(Wr23_fin / static_cast<double>(n_pairs));
  const MatrixXd Sigma_eta  = Wr23_inv * Sigma_W_eta * Wr23_inv
  / static_cast<double>(n);
  
  // Assemble Sigma_theta (Beta + Eta block)
  MatrixXd Sigma_theta_BE = MatrixXd::Zero(p_BE, p_BE);
  Sigma_theta_BE.block(0,      0,      p_Beta, p_Beta) = Sigma_beta;
  Sigma_theta_BE.block(p_Beta, p_Beta, p_Eta,  p_Eta)  = Sigma_eta;
  const VectorXd sigma_theta = Sigma_theta_BE.diagonal();
  
  // --- (e) Separate subject-level cluster-robust sandwich for Xi ---
  // s_i = sum_t g_{it} * (r_{it} - p_{it})   [subject i's logistic score]
  // B_xi_fin = (1/n) sum_i sum_t g_{it} g_{it}^T p_{it}(1-p_{it})
  // S_xi_fin = (1/n) sum_i s_i s_i^T
  // Sigma_xi = B_xi^{-1} S_xi B_xi^{-1} / n
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
  
  const double dn = static_cast<double>(n);
  const MatrixXd Bxi_inv = pinv_sym(Bxi_fin / dn);
  const MatrixXd Sigma_xi_logit = Bxi_inv * (Sxi_fin / dn) * Bxi_inv / dn;
  const VectorXd sigma_xi_logit = Sigma_xi_logit.diagonal();
  
  // --------------------------------------------------------
  // 6. Pack and return
  // --------------------------------------------------------
  NumericVector theta_out(p);
  for (int r_ = 0; r_ < p_Beta; ++r_) theta_out[r_]            = Beta(r_);
  for (int r_ = 0; r_ < p_Eta;  ++r_) theta_out[p_Beta + r_]   = Eta(r_);
  for (int r_ = 0; r_ < p_Xi;   ++r_) theta_out[p_BE   + r_]   = Xi(r_);
  
  return Rcpp::List::create(
    _["theta.est"]         = theta_out,
    _["Beta"]              = Rcpp::wrap(Beta),
    _["Eta"]               = Rcpp::wrap(Eta),
    _["Xi"]                = Rcpp::wrap(Xi),
    _["Sigma.theta"]       = Rcpp::wrap(Sigma_theta_BE),
    _["sigma.theta"]       = Rcpp::wrap(sigma_theta),
    _["Sigma.xi.logit"]    = Rcpp::wrap(Sigma_xi_logit),
    _["sigma.xi.logit"]    = Rcpp::wrap(sigma_xi_logit),
    _["iter"]              = iter,
    _["converged"]         = converged
  );
}