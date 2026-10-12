#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;

// ============================================================
// Helpers
// ============================================================

static inline double pospart(double x) {
  return (x > 0.0) ? x : 0.0;
}

static inline MatrixXd build_kvecs(const int K, const double t_star) {
  MatrixXd Z(K, 3);
  for (int k = 0; k < K; ++k) {
    const double kk = static_cast<double>(k + 1); // visits indexed 1,...,K
    Z(k, 0) = 1.0;
    Z(k, 1) = kk;
    Z(k, 2) = pospart(kk - t_star);
  }
  return Z;
}

// Variance-model design. basis 0 = knot (identical to the mean design),
// basis 1 = orthogonal quadratic (Gram-Schmidt on 1, k, k^2, unit mean square).
static inline MatrixXd build_zvar(const int K, const double t_star, const int basis) {
  if (basis == 0) return build_kvecs(K, t_star);
  MatrixXd Z(K, 3);
  VectorXd k1(K), k2(K);
  for (int k = 0; k < K; ++k) { double kk = k + 1.0; k1(k) = kk; k2(k) = kk * kk; }
  VectorXd c1 = k1 - VectorXd::Constant(K, k1.mean());
  VectorXd c2 = k2 - VectorXd::Constant(K, k2.mean());
  c2 -= (c1.dot(c2) / c1.dot(c1)) * c1;
  c1 /= std::sqrt(c1.dot(c1) / K);
  c2 /= std::sqrt(c2.dot(c2) / K);
  Z.col(0) = VectorXd::Ones(K); Z.col(1) = c1; Z.col(2) = c2;
  return Z;
}

static inline bool is_finite_vec(const VectorXd& x) {
  for (int i = 0; i < x.size(); ++i) {
    if (!R_finite(x(i))) return false;
  }
  return true;
}

static inline bool is_finite_mat(const MatrixXd& X) {
  for (int i = 0; i < X.rows(); ++i) {
    for (int j = 0; j < X.cols(); ++j) {
      if (!R_finite(X(i, j))) return false;
    }
  }
  return true;
}

static inline MatrixXd nearest_psd(const MatrixXd& M, const double eps = 1e-8) {
  MatrixXd S = 0.5 * (M + M.transpose());
  SelfAdjointEigenSolver<MatrixXd> es(S);
  if (es.info() != Eigen::Success) {
    MatrixXd Sr = S;
    Sr.diagonal().array() += 1e-6;
    return Sr;
  }
  VectorXd eval = es.eigenvalues();
  MatrixXd evec = es.eigenvectors();
  for (int i = 0; i < eval.size(); ++i) {
    eval(i) = std::max(eval(i), eps);
  }
  MatrixXd out = evec * eval.asDiagonal() * evec.transpose();
  out = 0.5 * (out + out.transpose());
  return out;
}

static inline double clip_exchangeable_rho(const int m, double rho, const double eps = 1e-6) {
  if (m <= 1) return 0.0;
  const double lo = -1.0 / static_cast<double>(m - 1) + eps;
  const double hi =  1.0 - eps;
  if (rho < lo) rho = lo;
  if (rho > hi) rho = hi;
  return rho;
}

static inline double clip_ar1_rho(double rho, const double eps = 1e-6) {
  const double lo = -1.0 + eps;
  const double hi =  1.0 - eps;
  if (rho < lo) rho = lo;
  if (rho > hi) rho = hi;
  return rho;
}

static inline MatrixXd build_corr_mat(const int m,
                                      const std::string& corr_type,
                                      double rho) {
  MatrixXd R = MatrixXd::Identity(m, m);
  if (m <= 1) return R;
  
  if (corr_type == "independence" || corr_type == "ind") {
    return R;
  }
  
  if (corr_type == "exchangeable" || corr_type == "exch") {
    rho = clip_exchangeable_rho(m, rho);
    R.setConstant(rho);
    R.diagonal().setOnes();
    return nearest_psd(R);
  }
  
  if (corr_type == "ar1" || corr_type == "AR1" || corr_type == "AR(1)") {
    rho = clip_ar1_rho(rho);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        R(i, j) = std::pow(rho, std::abs(i - j));
      }
    }
    return nearest_psd(R);
  }
  
  if (corr_type == "model") {
    return R; // not used; caller directly passes model covariance
  }
  
  stop("Unknown corr_type. Use 'independence', 'exchangeable', 'ar1', or 'model'.");
  return R;
}

static inline MatrixXd diag_to_cov_with_corr(const MatrixXd& Sigma_model,
                                             const std::string& corr_type,
                                             double rho) {
  const int m = Sigma_model.rows();
  MatrixXd Ahalf = MatrixXd::Zero(m, m);
  for (int i = 0; i < m; ++i) {
    Ahalf(i, i) = std::sqrt(std::max(Sigma_model(i, i), 1e-12));
  }
  MatrixXd R = build_corr_mat(m, corr_type, rho);
  MatrixXd V = Ahalf * R * Ahalf;
  V = 0.5 * (V + V.transpose());
  V.diagonal().array() += 1e-10;
  return nearest_psd(V);
}

static inline MatrixXd choose_working_cov(const MatrixXd& Sigma_model,
                                          const std::string& corr_type,
                                          double rho,
                                          const bool use_full_model_cov) {
  if ((use_full_model_cov && corr_type == "model") || corr_type == "model") {
    MatrixXd V = Sigma_model;
    V.diagonal().array() += 1e-10;
    return nearest_psd(V);
  }
  return diag_to_cov_with_corr(Sigma_model, corr_type, rho);
}

static inline void add_block_contrib_full(const MatrixXd& D,
                                          const VectorXd& S,
                                          const MatrixXd& V_in,
                                          VectorXd& U,
                                          MatrixXd& W) {
  MatrixXd V = nearest_psd(V_in);
  Eigen::LLT<MatrixXd> llt(V);
  
  if (llt.info() != Eigen::Success) {
    V.diagonal().array() += 1e-6;
    V = nearest_psd(V);
    llt.compute(V);
    if (llt.info() != Eigen::Success) {
      Eigen::LDLT<MatrixXd> ldlt(V);
      if (ldlt.info() != Eigen::Success) {
        stop("Working covariance is not positive definite, even after repair.");
      }
      VectorXd VinvS = ldlt.solve(S);
      MatrixXd VinvD = ldlt.solve(D);
      U.noalias() += D.transpose() * VinvS;
      W.noalias() += D.transpose() * VinvD;
      return;
    }
  }
  
  VectorXd VinvS = llt.solve(S);
  MatrixXd VinvD = llt.solve(D);
  
  U.noalias() += D.transpose() * VinvS;
  W.noalias() += D.transpose() * VinvD;
}

static inline VectorXd safe_solve(const MatrixXd& A, const VectorXd& b, const double ridge = 1e-8) {
  MatrixXd Ar = 0.5 * (A + A.transpose());
  Ar.diagonal().array() += ridge;
  Eigen::LDLT<MatrixXd> ldlt(Ar);
  if (ldlt.info() == Eigen::Success) {
    return ldlt.solve(b);
  }
  Ar = nearest_psd(Ar, 1e-8);
  Ar.diagonal().array() += 1e-6;
  Eigen::LDLT<MatrixXd> ldlt2(Ar);
  if (ldlt2.info() != Eigen::Success) {
    stop("Matrix solve failed.");
  }
  return ldlt2.solve(b);
}

static inline double covX(const int a,
                          const int b,
                          const double sigmaD2,
                          const VectorXd& xi) {
  if (a == b) return sigmaD2 + 2.0 * xi(a);
  return sigmaD2;
}

static inline MatrixXd block1_model_cov(const double sigma_sum,
                                        const VectorXd& xi) {
  const int K = xi.size();
  MatrixXd V(K, K);
  for (int a = 0; a < K; ++a) {
    for (int b = 0; b < K; ++b) {
      V(a, b) = 0.25 * sigma_sum + ((a == b) ? 0.5 * xi(a) : 0.0);
    }
  }
  return nearest_psd(V);
}

static inline MatrixXd block2_model_cov(const double sigmaD2,
                                        const VectorXd& xi) {
  const int K = xi.size();
  const int m = K - 1;
  MatrixXd V(m, m);
  for (int a = 0; a < m; ++a) {
    for (int b = 0; b < m; ++b) {
      const double s1 = covX(a,     b,     sigmaD2, xi);
      const double s2 = covX(a + 1, b + 1, sigmaD2, xi);
      const double s3 = covX(a,     b + 1, sigmaD2, xi);
      const double s4 = covX(a + 1, b,     sigmaD2, xi);
      V(a, b) = 0.25 * (s1 * s2 + s3 * s4);
    }
  }
  return nearest_psd(V);
}

static inline MatrixXd block3_model_cov(const double sigmaD2,
                                        const VectorXd& xi) {
  const int K = xi.size();
  MatrixXd V(K, K);
  for (int a = 0; a < K; ++a) {
    for (int b = 0; b < K; ++b) {
      const double s = covX(a, b, sigmaD2, xi);
      V(a, b) = 0.5 * s * s;
    }
  }
  return nearest_psd(V);
}

static inline void compute_sigb_xi(const VectorXd& Eta,
                                   const MatrixXd& Z,
                                   const VectorXd& u,
                                   VectorXd& sigb,
                                   VectorXd& xi) {
  const int n = u.size();
  const int K = Z.rows();
  
  sigb.resize(n);
  xi.resize(K);
  
  for (int i = 0; i < n; ++i) {
    double val = Eta(0) + Eta(1) * u(i);
    val = std::min(30.0, std::max(-30.0, val));
    sigb(i) = std::exp(val);
  }
  for (int k = 0; k < K; ++k) {
    double val = Eta.segment<3>(2).dot(Z.row(k));
    val = std::min(30.0, std::max(-30.0, val));
    xi(k) = std::exp(val);
  }
}

static inline MatrixXd compute_residuals(const MatrixXd& Y,
                                         const MatrixXd& Z,
                                         const VectorXd& beta) {
  return Y.rowwise() - (Z * beta).transpose();
}

static inline void initialize_beta_eta_moments(
    const MatrixXd& Y,
    const VectorXd& u,
    const MatrixXd& Z,
    const MatrixXd& Zv,
    VectorXd& beta,
    VectorXd& Eta
) {
  const int n = Y.rows();
  const int K = Y.cols();
  
  // OLS beta on stacked data
  MatrixXd Xbig(n * K, 3);
  VectorXd ybig(n * K);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < K; ++k) {
      Xbig.row(i * K + k) = Z.row(k);
      ybig(i * K + k) = Y(i, k);
    }
  }
  beta = (Xbig.transpose() * Xbig).ldlt().solve(Xbig.transpose() * ybig);
  
  // residuals
  MatrixXd E0 = compute_residuals(Y, Z, beta);
  
  // subject means -> between-subject variation
  VectorXd subj_mean(n);
  for (int i = 0; i < n; ++i) {
    subj_mean(i) = E0.row(i).mean();
  }
  
  MatrixXd Xu(n, 2);
  Xu.col(0).setOnes();
  Xu.col(1) = u;
  
  VectorXd y_alpha(n);
  const double c_alpha = 1e-4;
  for (int i = 0; i < n; ++i) {
    y_alpha(i) = std::log(subj_mean(i) * subj_mean(i) + c_alpha);
  }
  VectorXd alpha_init = (Xu.transpose() * Xu).ldlt().solve(Xu.transpose() * y_alpha);
  
  // demean within subject -> within-subject variation
  MatrixXd Ew = E0;
  for (int i = 0; i < n; ++i) {
    const double mi = E0.row(i).mean();
    Ew.row(i).array() -= mi;
  }
  
  VectorXd logv(K);
  for (int k = 0; k < K; ++k) {
    double vk = Ew.col(k).array().square().mean();
    vk = std::max(vk, 1e-4);
    logv(k) = std::log(vk);
  }
  VectorXd tau_init = (Zv.transpose() * Zv).ldlt().solve(Zv.transpose() * logv);
  
  Eta.resize(5);
  Eta(0) = std::max(-10.0, std::min(10.0, alpha_init(0)));
  Eta(1) = std::max(-5.0,  std::min(5.0,  alpha_init(1)));
  Eta(2) = std::max(-10.0, std::min(10.0, tau_init(0)));
  Eta(3) = std::max(-10.0, std::min(10.0, tau_init(1)));
  Eta(4) = std::max(-10.0, std::min(10.0, tau_init(2)));
}

static inline double estimate_rho_from_stdres(const std::string& corr_type,
                                              const std::vector<VectorXd>& Zres,
                                              double rho_old) {
  if (corr_type == "independence" || corr_type == "ind" || corr_type == "model") {
    return 0.0;
  }
  
  double num = 0.0;
  double den = 0.0;
  
  if (corr_type == "exchangeable" || corr_type == "exch") {
    for (size_t b = 0; b < Zres.size(); ++b) {
      const VectorXd& z = Zres[b];
      const int m = z.size();
      if (m <= 1) continue;
      for (int i = 0; i < m - 1; ++i) {
        for (int j = i + 1; j < m; ++j) {
          num += z(i) * z(j);
          den += 1.0;
        }
      }
    }
    if (den <= 0.0) return rho_old;
    return num / den;
  }
  
  if (corr_type == "ar1" || corr_type == "AR1" || corr_type == "AR(1)") {
    for (size_t b = 0; b < Zres.size(); ++b) {
      const VectorXd& z = Zres[b];
      const int m = z.size();
      if (m <= 1) continue;
      for (int i = 0; i < m - 1; ++i) {
        num += z(i) * z(i + 1);
        den += 1.0;
      }
    }
    if (den <= 0.0) return rho_old;
    return num / den;
  }
  
  return rho_old;
}

static inline double clip_rho_by_type(const std::string& corr_type,
                                      const int m,
                                      const double rho) {
  if (corr_type == "exchangeable" || corr_type == "exch") {
    return clip_exchangeable_rho(m, rho);
  }
  if (corr_type == "ar1" || corr_type == "AR1" || corr_type == "AR(1)") {
    return clip_ar1_rho(rho);
  }
  return 0.0;
}

static inline bool params_are_safe(const VectorXd& beta,
                                   const VectorXd& Eta) {
  if (!is_finite_vec(beta) || !is_finite_vec(Eta)) return false;
  for (int i = 0; i < Eta.size(); ++i) {
    if (std::abs(Eta(i)) > 30.0) return false;
  }
  return true;
}

static inline double max_abs_change(const VectorXd& a, const VectorXd& b) {
  return (a - b).cwiseAbs().maxCoeff();
}

static inline bool is_ind_or_model(const std::string& corr_type) {
  return (corr_type == "independence" || corr_type == "ind" || corr_type == "model");
}

static inline std::string effective_corr_block2(const std::string& corr2,
                                                const bool use_full_model_cov) {
  return corr2;  // PATCHED: no silent block-2 override
}

static inline void update_rho_relaxed(double& rho,
                                      const std::string& corr_type,
                                      const int m,
                                      const std::vector<VectorXd>& zres,
                                      const double relax) {
  if (is_ind_or_model(corr_type)) {
    rho = 0.0;
    return;
  }
  const double rho_new = estimate_rho_from_stdres(corr_type, zres, rho);
  rho = (1.0 - relax) * rho + relax * rho_new;
  rho = clip_rho_by_type(corr_type, m, rho);
}

// ============================================================
// Main UGEE with full model-based block covariance
// ============================================================

// [[Rcpp::export]]
Rcpp::List Dist_UGEE_split_cpp(
    const Eigen::Map<Eigen::MatrixXd> Y,   // n x K
    const Eigen::Map<Eigen::VectorXd> u,   // length n
    const double t_star,
    const int var_basis = 0,          // 0 = knot (as before), 1 = orthogonal quadratic
    const std::string corr1 = "model",
    const std::string corr2 = "model",
    const std::string corr3 = "model",
    double rho1 = 0.0,
    double rho2 = 0.0,
    double rho3 = 0.0,
    const bool estimate_rho = true,
    const bool use_full_model_cov = true,
    const int maxit = 100,
    const double tol = 1e-8,
    const double max_step = 1.0,
    const bool verbose = false,
    Rcpp::Nullable<Rcpp::NumericVector> beta_init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> Eta_init  = R_NilValue,
    const double ridge = 1e-6,
    const double rho_relax = 0.3
) {
  const int n = Y.rows();
  const int K = Y.cols();
  
  if (u.size() != n) stop("Length of u must equal number of rows of Y.");
  if (K < 2) stop("Need at least K >= 2 visits.");
  
  const int p_beta = 3;
  const int p_eta  = 5;
  
  MatrixXd Z  = build_kvecs(K, t_star);          // MEAN design (knot)
  MatrixXd Zv = build_zvar(K, t_star, var_basis);  // VARIANCE design
  const std::string corr2_eff = effective_corr_block2(corr2, use_full_model_cov);
  
  // ------------------------------------------------------------
  // Initialize beta, Eta
  // ------------------------------------------------------------
  VectorXd beta(p_beta);
  VectorXd Eta(p_eta);
  
  VectorXd beta_def, Eta_def;
  initialize_beta_eta_moments(Y, u, Z, Zv, beta_def, Eta_def);
  
  if (beta_init.isNotNull()) {
    NumericVector b0(beta_init);
    if (b0.size() != p_beta) stop("beta_init must have length 3.");
    for (int j = 0; j < p_beta; ++j) beta(j) = b0[j];
  } else {
    beta = beta_def;
  }
  
  if (Eta_init.isNotNull()) {
    NumericVector e0(Eta_init);
    if (e0.size() != p_eta) stop("Eta_init must have length 5.");
    for (int j = 0; j < p_eta; ++j) Eta(j) = e0[j];
  } else {
    Eta = Eta_def;
  }
  
  if (!params_are_safe(beta, Eta)) {
    beta = beta_def;
    Eta  = Eta_def;
  }
  
  int iter = 0;
  double max_change = NA_REAL;
  
  // ============================================================
  // Newton-Raphson iterations with damping
  // ============================================================
  for (iter = 0; iter < maxit; ++iter) {
    VectorXd sigb, xi;
    compute_sigb_xi(Eta, Zv, u, sigb, xi);
    MatrixXd E = compute_residuals(Y, Z, beta);
    
    // ----------------------------------------------------------
    // Step 0: estimate working correlations from standardized residuals
    // ----------------------------------------------------------
    if (estimate_rho) {
      std::vector<VectorXd> zres1;
      std::vector<VectorXd> zres2;
      std::vector<VectorXd> zres3;
      zres1.reserve(n * (n - 1) / 2);
      zres2.reserve(n * (n - 1) / 2);
      zres3.reserve(n * (n - 1) / 2);
      
      for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
          const double sigma_sum = sigb(i) + sigb(j);
          const double sigmaD2   = sigma_sum;
          const double h2_pair   = 0.5 * sigmaD2;
          
          // block 1
          {
            MatrixXd V1m = block1_model_cov(sigma_sum, xi);
            VectorXd sd = V1m.diagonal().array().sqrt();
            VectorXd S1(K);
            for (int k = 0; k < K; ++k) {
              const double f1 = 0.5 * (Y(i, k) + Y(j, k));
              const double h1 = beta.dot(Z.row(k));
              S1(k) = (f1 - h1) / std::max(sd(k), 1e-12);
            }
            zres1.push_back(S1);
          }
          
          // block 2
          {
            MatrixXd V2m = block2_model_cov(sigmaD2, xi);
            VectorXd sd = V2m.diagonal().array().sqrt();
            VectorXd S2(K - 1);
            for (int k = 0; k < K - 1; ++k) {
              const double d1 = E(i, k)     - E(j, k);
              const double d2 = E(i, k + 1) - E(j, k + 1);
              const double f2 = 0.5 * d1 * d2;
              S2(k) = (f2 - h2_pair) / std::max(sd(k), 1e-12);
            }
            zres2.push_back(S2);
          }
          
          // block 3
          {
            MatrixXd V3m = block3_model_cov(sigmaD2, xi);
            VectorXd sd = V3m.diagonal().array().sqrt();
            VectorXd S3(K);
            for (int k = 0; k < K; ++k) {
              const double d = E(i, k) - E(j, k);
              const double f3 = 0.5 * d * d;
              const double h3 = h2_pair + xi(k);
              S3(k) = (f3 - h3) / std::max(sd(k), 1e-12);
            }
            zres3.push_back(S3);
          }
        }
      }
      
      update_rho_relaxed(rho1, corr1, K,     zres1, rho_relax);
      update_rho_relaxed(rho2, corr2_eff, K - 1, zres2, rho_relax);
      update_rho_relaxed(rho3, corr3, K,     zres3, rho_relax);
    }
    
    // ----------------------------------------------------------
    // Step 1: update beta using block 1
    // ----------------------------------------------------------
    VectorXd U1 = VectorXd::Zero(p_beta);
    MatrixXd W1 = MatrixXd::Zero(p_beta, p_beta);
    
    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double sigma_sum = sigb(i) + sigb(j);
        MatrixXd V1m = block1_model_cov(sigma_sum, xi);
        MatrixXd V1  = choose_working_cov(V1m, corr1, rho1, use_full_model_cov);
        
        VectorXd S1(K);
        MatrixXd D1(K, p_beta);
        for (int k = 0; k < K; ++k) {
          const double f1 = 0.5 * (Y(i, k) + Y(j, k));
          const double h1 = beta.dot(Z.row(k));
          S1(k) = f1 - h1;
          D1.row(k) = Z.row(k);
        }
        
        add_block_contrib_full(D1, S1, V1, U1, W1);
      }
    }
    
    W1.diagonal().array() += ridge;
    VectorXd step_beta = safe_solve(W1, U1, ridge);
    
    // ----------------------------------------------------------
    // Step 2: update Eta using blocks 2 and 3
    // ----------------------------------------------------------
    VectorXd U23 = VectorXd::Zero(p_eta);
    MatrixXd W23 = MatrixXd::Zero(p_eta, p_eta);
    
    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double sigmaD2 = sigb(i) + sigb(j);
        const double h2_pair = 0.5 * sigmaD2;
        const double dh2_da0 = h2_pair;
        const double dh2_da1 = 0.5 * (u(i) * sigb(i) + u(j) * sigb(j));
        
        // block 2
        {
          MatrixXd V2m = block2_model_cov(sigmaD2, xi);
          MatrixXd V2  = choose_working_cov(V2m, corr2_eff, rho2, use_full_model_cov);
          
          VectorXd S2(K - 1);
          MatrixXd D2(K - 1, p_eta);
          D2.setZero();
          
          for (int k = 0; k < K - 1; ++k) {
            const double d1 = E(i, k)     - E(j, k);
            const double d2 = E(i, k + 1) - E(j, k + 1);
            const double f2 = 0.5 * d1 * d2;
            
            S2(k) = f2 - h2_pair;
            D2(k, 0) = dh2_da0;
            D2(k, 1) = dh2_da1;
          }
          
          add_block_contrib_full(D2, S2, V2, U23, W23);
        }
        
        // block 3
        {
          MatrixXd V3m = block3_model_cov(sigmaD2, xi);
          MatrixXd V3  = choose_working_cov(V3m, corr3, rho3, use_full_model_cov);
          
          VectorXd S3(K);
          MatrixXd D3(K, p_eta);
          D3.setZero();
          
          for (int k = 0; k < K; ++k) {
            const double d = E(i, k) - E(j, k);
            const double f3 = 0.5 * d * d;
            const double h3 = h2_pair + xi(k);
            
            S3(k) = f3 - h3;
            
            D3(k, 0) = dh2_da0;
            D3(k, 1) = dh2_da1;
            D3(k, 2) = xi(k) * Zv(k, 0);
            D3(k, 3) = xi(k) * Zv(k, 1);
            D3(k, 4) = xi(k) * Zv(k, 2);
          }
          
          add_block_contrib_full(D3, S3, V3, U23, W23);
        }
      }
    }
    
    W23.diagonal().array() += ridge;
    VectorXd step_eta = safe_solve(W23, U23, ridge);
    
    // ----------------------------------------------------------
    // Damping / backtracking
    // ----------------------------------------------------------
    double lambda = 1.0;
    const double step_norm =
      std::max(step_beta.cwiseAbs().maxCoeff(), step_eta.cwiseAbs().maxCoeff());
    
    if (step_norm > max_step) {
      lambda = max_step / step_norm;
    }
    
    VectorXd beta_new = beta;
    VectorXd Eta_new  = Eta;
    bool accepted = false;
    
    for (int bt = 0; bt < 20; ++bt) {
      beta_new = beta + lambda * step_beta;
      Eta_new  = Eta  + lambda * step_eta;
      
      if (!params_are_safe(beta_new, Eta_new)) {
        lambda *= 0.5;
        continue;
      }
      
      VectorXd sigb_try, xi_try;
      compute_sigb_xi(Eta_new, Zv, u, sigb_try, xi_try);
      
      if (!(is_finite_vec(sigb_try) && is_finite_vec(xi_try))) {
        lambda *= 0.5;
        continue;
      }
      
      const int jrep = std::min(1, n - 1);
      const double sigmaD2_try = sigb_try(0) + sigb_try(jrep);
      MatrixXd V2try = choose_working_cov(block2_model_cov(sigmaD2_try, xi_try),
                                          corr2_eff, rho2, use_full_model_cov);
      MatrixXd V3try = choose_working_cov(block3_model_cov(sigmaD2_try, xi_try),
                                          corr3, rho3, use_full_model_cov);
      
      if (!is_finite_mat(V2try) || !is_finite_mat(V3try)) {
        lambda *= 0.5;
        continue;
      }
      
      accepted = true;
      break;
    }
    
    if (!accepted) {
      warning("Backtracking failed to find a stable Newton step; stopping early.");
      break;
    }
    
    max_change = std::max(max_abs_change(beta_new, beta),
                          max_abs_change(Eta_new, Eta));
    
    beta = beta_new;
    Eta  = Eta_new;
    
    if (verbose) {
      Rcout << "iter = " << iter + 1
            << ", max_change = " << max_change
            << ", rho1 = " << rho1
            << ", rho2 = " << rho2
            << ", rho3 = " << rho3
            << "\n";
    }
    
    if (max_change < tol) break;
  }
  
  // ============================================================
  // Final quantities
  // ============================================================
  VectorXd sigb, xi;
  compute_sigb_xi(Eta, Zv, u, sigb, xi);
  MatrixXd E = compute_residuals(Y, Z, beta);
  
  // Final rho update
  if (estimate_rho) {
    std::vector<VectorXd> zres1;
    std::vector<VectorXd> zres2;
    std::vector<VectorXd> zres3;
    zres1.reserve(n * (n - 1) / 2);
    zres2.reserve(n * (n - 1) / 2);
    zres3.reserve(n * (n - 1) / 2);
    
    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double sigma_sum = sigb(i) + sigb(j);
        const double sigmaD2   = sigma_sum;
        const double h2_pair   = 0.5 * sigmaD2;
        
        // block 1
        {
          MatrixXd V1m = block1_model_cov(sigma_sum, xi);
          VectorXd sd = V1m.diagonal().array().sqrt();
          VectorXd S1(K);
          for (int k = 0; k < K; ++k) {
            const double f1 = 0.5 * (Y(i, k) + Y(j, k));
            const double h1 = beta.dot(Z.row(k));
            S1(k) = (f1 - h1) / std::max(sd(k), 1e-12);
          }
          zres1.push_back(S1);
        }
        
        // block 2
        {
          MatrixXd V2m = block2_model_cov(sigmaD2, xi);
          VectorXd sd = V2m.diagonal().array().sqrt();
          VectorXd S2(K - 1);
          for (int k = 0; k < K - 1; ++k) {
            const double d1 = E(i, k)     - E(j, k);
            const double d2 = E(i, k + 1) - E(j, k + 1);
            const double f2 = 0.5 * d1 * d2;
            S2(k) = (f2 - h2_pair) / std::max(sd(k), 1e-12);
          }
          zres2.push_back(S2);
        }
        
        // block 3
        {
          MatrixXd V3m = block3_model_cov(sigmaD2, xi);
          VectorXd sd = V3m.diagonal().array().sqrt();
          VectorXd S3(K);
          for (int k = 0; k < K; ++k) {
            const double d = E(i, k) - E(j, k);
            const double f3 = 0.5 * d * d;
            const double h3 = h2_pair + xi(k);
            S3(k) = (f3 - h3) / std::max(sd(k), 1e-12);
          }
          zres3.push_back(S3);
        }
      }
    }
    
    rho1 = is_ind_or_model(corr1) ? 0.0 : clip_rho_by_type(corr1, K,     estimate_rho_from_stdres(corr1, zres1, rho1));
    rho2 = is_ind_or_model(corr2_eff) ? 0.0 : clip_rho_by_type(corr2_eff, K - 1, estimate_rho_from_stdres(corr2_eff, zres2, rho2));
    rho3 = is_ind_or_model(corr3) ? 0.0 : clip_rho_by_type(corr3, K,     estimate_rho_from_stdres(corr3, zres3, rho3));
  }
  
  // ============================================================
  // Final bread blocks
  // ============================================================
  MatrixXd B1  = MatrixXd::Zero(p_beta, p_beta);
  MatrixXd B23 = MatrixXd::Zero(p_eta, p_eta);
  
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double sigma_sum = sigb(i) + sigb(j);
      const double sigmaD2   = sigma_sum;
      const double h2_pair   = 0.5 * sigmaD2;
      const double dh2_da0   = h2_pair;
      const double dh2_da1   = 0.5 * (u(i) * sigb(i) + u(j) * sigb(j));
      
      // block 1
      {
        MatrixXd D1(K, p_beta);
        for (int k = 0; k < K; ++k) D1.row(k) = Z.row(k);
        MatrixXd V1 = choose_working_cov(block1_model_cov(sigma_sum, xi),
                                         corr1, rho1, use_full_model_cov);
        VectorXd dummyU = VectorXd::Zero(p_beta);
        VectorXd zeroS  = VectorXd::Zero(K);
        add_block_contrib_full(D1, zeroS, V1, dummyU, B1);
      }
      
      // block 2
      {
        MatrixXd D2(K - 1, p_eta);
        D2.setZero();
        for (int k = 0; k < K - 1; ++k) {
          D2(k, 0) = dh2_da0;
          D2(k, 1) = dh2_da1;
        }
        MatrixXd V2 = choose_working_cov(block2_model_cov(sigmaD2, xi),
                                         corr2_eff, rho2, use_full_model_cov);
        VectorXd dummyU = VectorXd::Zero(p_eta);
        VectorXd zeroS  = VectorXd::Zero(K - 1);
        add_block_contrib_full(D2, zeroS, V2, dummyU, B23);
      }
      
      // block 3
      {
        MatrixXd D3(K, p_eta);
        D3.setZero();
        for (int k = 0; k < K; ++k) {
          D3(k, 0) = dh2_da0;
          D3(k, 1) = dh2_da1;
          D3(k, 2) = xi(k) * Zv(k, 0);
          D3(k, 3) = xi(k) * Zv(k, 1);
          D3(k, 4) = xi(k) * Zv(k, 2);
        }
        MatrixXd V3 = choose_working_cov(block3_model_cov(sigmaD2, xi),
                                         corr3, rho3, use_full_model_cov);
        VectorXd dummyU = VectorXd::Zero(p_eta);
        VectorXd zeroS  = VectorXd::Zero(K);
        add_block_contrib_full(D3, zeroS, V3, dummyU, B23);
      }
    }
  }
  
  const double npairs = static_cast<double>(n * (n - 1) / 2);
  B1  /= npairs;
  B23 /= npairs;
  
  // ============================================================
  // Hájek-style meat approximation
  // ============================================================
  MatrixXd meat_beta = MatrixXd::Zero(p_beta, p_beta);
  MatrixXd meat_eta  = MatrixXd::Zero(p_eta, p_eta);
  
  for (int i = 0; i < n; ++i) {
    VectorXd U_beta_i = VectorXd::Zero(p_beta);
    VectorXd U_eta_i  = VectorXd::Zero(p_eta);
    
    for (int j = 0; j < n; ++j) {
      if (j == i) continue;
      
      const double sigma_sum = sigb(i) + sigb(j);
      const double sigmaD2   = sigma_sum;
      const double h2_pair   = 0.5 * sigmaD2;
      const double dh2_da0   = h2_pair;
      const double dh2_da1   = 0.5 * (u(i) * sigb(i) + u(j) * sigb(j));
      
      // block 1
      {
        MatrixXd V1 = choose_working_cov(block1_model_cov(sigma_sum, xi),
                                         corr1, rho1, use_full_model_cov);
        
        VectorXd S1(K);
        MatrixXd D1(K, p_beta);
        for (int k = 0; k < K; ++k) {
          const double f1 = 0.5 * (Y(i, k) + Y(j, k));
          const double h1 = beta.dot(Z.row(k));
          S1(k) = f1 - h1;
          D1.row(k) = Z.row(k);
        }
        
        MatrixXd dummyW = MatrixXd::Zero(p_beta, p_beta);
        add_block_contrib_full(D1, S1, V1, U_beta_i, dummyW);
      }
      
      // block 2
      {
        MatrixXd V2 = choose_working_cov(block2_model_cov(sigmaD2, xi),
                                         corr2_eff, rho2, use_full_model_cov);
        
        VectorXd S2(K - 1);
        MatrixXd D2(K - 1, p_eta);
        D2.setZero();
        
        for (int k = 0; k < K - 1; ++k) {
          const double d1 = E(i, k)     - E(j, k);
          const double d2 = E(i, k + 1) - E(j, k + 1);
          const double f2 = 0.5 * d1 * d2;
          
          S2(k) = f2 - h2_pair;
          D2(k, 0) = dh2_da0;
          D2(k, 1) = dh2_da1;
        }
        
        MatrixXd dummyW = MatrixXd::Zero(p_eta, p_eta);
        add_block_contrib_full(D2, S2, V2, U_eta_i, dummyW);
      }
      
      // block 3
      {
        MatrixXd V3 = choose_working_cov(block3_model_cov(sigmaD2, xi),
                                         corr3, rho3, use_full_model_cov);
        
        VectorXd S3(K);
        MatrixXd D3(K, p_eta);
        D3.setZero();
        
        for (int k = 0; k < K; ++k) {
          const double d = E(i, k) - E(j, k);
          const double f3 = 0.5 * d * d;
          const double h3 = h2_pair + xi(k);
          
          S3(k) = f3 - h3;
          D3(k, 0) = dh2_da0;
          D3(k, 1) = dh2_da1;
          D3(k, 2) = xi(k) * Zv(k, 0);
          D3(k, 3) = xi(k) * Zv(k, 1);
          D3(k, 4) = xi(k) * Zv(k, 2);
        }
        
        MatrixXd dummyW = MatrixXd::Zero(p_eta, p_eta);
        add_block_contrib_full(D3, S3, V3, U_eta_i, dummyW);
      }
    }
    
    U_beta_i /= static_cast<double>(n - 1);
    U_eta_i  /= static_cast<double>(n - 1);
    
    meat_beta.noalias() += U_beta_i * U_beta_i.transpose();
    meat_eta.noalias()  += U_eta_i  * U_eta_i.transpose();
  }
  
  meat_beta *= 4.0 / static_cast<double>(n);
  meat_eta  *= 4.0 / static_cast<double>(n);
  
  B1  = 0.5 * (B1 + B1.transpose());
  B23 = 0.5 * (B23 + B23.transpose());
  B1.diagonal().array()  += ridge;
  B23.diagonal().array() += ridge;
  
  MatrixXd Ibeta = MatrixXd::Identity(p_beta, p_beta);
  MatrixXd Ieta  = MatrixXd::Identity(p_eta, p_eta);
  MatrixXd B1_inv  = safe_solve(B1, Ibeta.col(0), ridge) * RowVectorXd::Zero(p_beta); // dummy init
  B1_inv.resize(p_beta, p_beta);
  MatrixXd B23_inv(p_eta, p_eta);
  
  {
    Eigen::LDLT<MatrixXd> ldlt1(B1);
    if (ldlt1.info() == Eigen::Success) {
      B1_inv = ldlt1.solve(Ibeta);
    } else {
      MatrixXd B1r = nearest_psd(B1);
      B1r.diagonal().array() += 1e-6;
      Eigen::LDLT<MatrixXd> ldlt1b(B1r);
      if (ldlt1b.info() != Eigen::Success) stop("Failed to invert bread_beta.");
      B1_inv = ldlt1b.solve(Ibeta);
    }
  }
  {
    Eigen::LDLT<MatrixXd> ldlt2(B23);
    if (ldlt2.info() == Eigen::Success) {
      B23_inv = ldlt2.solve(Ieta);
    } else {
      MatrixXd B23r = nearest_psd(B23);
      B23r.diagonal().array() += 1e-6;
      Eigen::LDLT<MatrixXd> ldlt2b(B23r);
      if (ldlt2b.info() != Eigen::Success) stop("Failed to invert bread_eta.");
      B23_inv = ldlt2b.solve(Ieta);
    }
  }
  
  MatrixXd vcov_beta = (1.0 / n) * B1_inv * meat_beta * B1_inv;
  MatrixXd vcov_eta  = (1.0 / n) * B23_inv * meat_eta * B23_inv;
  
  Rcpp::List out = Rcpp::List::create(
    _["beta"]               = beta,
    _["Eta"]                = Eta,
    _["alpha"]              = Eta.head(2),
    _["tau"]                = Eta.tail(3),
    _["xi"]                 = xi,
    _["sigb"]               = sigb,
    _["rho1"]               = rho1,
    _["rho2"]               = rho2,
    _["rho3"]               = rho3,
    _["bread_beta"]         = B1,
    _["bread_eta"]          = B23,
    _["meat_beta"]          = meat_beta,
    _["meat_eta"]           = meat_eta,
    _["vcov_beta"]          = vcov_beta,
    _["vcov_eta"]           = vcov_eta,
    _["iterations"]         = iter + 1,
    _["converged"]          = (R_finite(max_change) && max_change < tol),
    _["max_change"]         = max_change
  );
  out["use_full_model_cov"] = use_full_model_cov;
  out["estimate_rho"]       = estimate_rho;
  out["ridge"]              = ridge;
  out["rho_relax"]          = rho_relax;
  out["var_basis"]          = var_basis;
  out["Zvar"]               = Zv;
  out["corr1"]              = corr1;
  out["corr2"]              = corr2;
  out["corr2_eff"]          = corr2_eff;
  out["corr3"]              = corr3;
  return out;
}
