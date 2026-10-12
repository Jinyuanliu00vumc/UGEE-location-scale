## Production DGP. Knot mean (t*=2), orthogonal-quadratic variance. Design 2 / SPEC 2026-09-26.
ortho_quad <- function(K) {
  k <- 1:K; c1 <- k - mean(k); k2 <- k^2
  c2 <- k2 - mean(k2); c2 <- c2 - sum(c1*c2)/sum(c1*c1) * c1
  Z <- cbind(1, c1/sqrt(sum(c1^2)/K), c2/sqrt(sum(c2^2)/K))
  stopifnot(abs(sum(Z[,2]*Z[,3])) < 1e-10, abs(mean(Z[,2])) < 1e-10, abs(mean(Z[,3])) < 1e-10)
  colnames(Z) <- c("one","c1","c2"); Z
}
## tau_1 set per K so that R_WS = 5 throughout; tau_0 = -0.267, tau_2 = 0.30
tau_for_K <- function(K, tau0 = -0.267, tau2 = 0.30, R_WS = 5) {
  Z <- ortho_quad(K); c(tau0, log(R_WS)/(Z[K,2] - Z[1,2]), tau2)
}
BETA <- c(1, -2, 1); ALPHA <- c(log(0.4), 1.0); SIGMA_W <- 0.2
NU_B <- 4; NU_E <- 5; T_STAR <- 2; P_U <- 0.5

sim_prod <- function(N, K = 5, errors = c("gaussian","nonnormal"),
                     Beta = BETA, Alpha = ALPHA, Tau = NULL,
                     sigma_omega = SIGMA_W, p_u = P_U, t_star = T_STAR,
                     nu_b = NU_B, nu_e = NU_E, seed = NULL) {
  errors <- match.arg(errors); if (!is.null(seed)) set.seed(seed)
  if (is.null(Tau)) Tau <- tau_for_K(K)
  stopifnot(length(Beta)==3, length(Alpha)==2, length(Tau)==3, t_star < K, N >= 2)
  k <- 1:K; kp <- pmax(k - t_star, 0); Zv <- ortho_quad(K)
  u <- rbinom(N, 1L, p_u)
  omega <- rnorm(N, mean = -0.5*sigma_omega^2, sd = sigma_omega)   # E{exp(omega)} = 1
  stopifnot(abs(mean(exp(omega)) - 1) < 0.2)
  s2b <- exp(Alpha[1] + Alpha[2]*u)
  lxi <- as.vector(Zv %*% Tau)                 # marginal log within-subject variance
  s2e <- exp(outer(omega, lxi, "+"))
  stopifnot(all(is.finite(s2b)), all(is.finite(s2e)), all(s2e > 0))
  if (errors == "gaussian") {
    b <- rnorm(N, 0, sqrt(s2b)); E <- matrix(rnorm(N*K), N, K) * sqrt(s2e)
  } else {
    Xb <- rchisq(N, nu_b); b <- sqrt(s2b)*(Xb - nu_b)/sqrt(2*nu_b)
    E  <- matrix(rt(N*K, nu_e), N, K) * sqrt(s2e) * sqrt((nu_e-2)/nu_e)
  }
  mu <- Beta[1] + Beta[2]*k + Beta[3]*kp       # KNOT mean
  Y  <- matrix(mu, N, K, byrow = TRUE) + b + E
  stopifnot(all(is.finite(Y)))
  list(Y = Y, u = as.numeric(u), K = K, Zv = Zv, Tau = Tau, lxi = lxi,
       truth = c(Beta, Alpha, Tau),
       long = data.frame(id = rep(1:N, K), k = rep(k, each=N), post = rep(kp, each=N),
                         c1 = rep(Zv[,2], each=N), c2 = rep(Zv[,3], each=N),
                         u = rep(u, K), y = as.vector(Y)))
}
