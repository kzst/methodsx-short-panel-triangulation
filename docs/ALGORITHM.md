# Algorithm and computational contract

This document makes the MethodsX workflow reimplementable at the level of inputs, outputs, parameters, gates, stopping rules, and computational scaling.

## Inputs

Required panel information: ordered times `t=1,...,T`; repeated units `i=1,...,N`; target `y_it`; predictors `x_kit`, `k=1,...,K`; declared outcome-predictor-direction testing families; maximum lag `p_max`; primary and sensitivity common-shock controls; raw alpha, family q, Bayes-factor threshold, RF settings, placebo/shadow counts, and replication rules.

## Required edge-level outputs

Retain all lag-specific classical p-values; minimum raw and lag-search-adjusted p-values; BY q-value (primary) and BH q-value (sensitivity); Bayesian best lag and `log(BF10)`; RF forward MSE gains, positive-fold share, shadow Q0.95 and empirical shadow p-value; unit/layer replication counts; placebo and common-shock sensitivity; nominal method indicators; linear/predictive family indicators; and final tier. Unsupported and not-testable edges remain in the audit table.

## Reference parameters

`p_max=2`; raw `alpha=.01`; BY `q=.05` primary and BH `q=.05` sensitivity; `BF10>=5` evaluated on the log scale; RF reference 500 trees and 20 circular-shift shadows; positive-fold-share convention .67; one-step forward validation; no imputation by default; full time effects primary with lagged leave-one-unit-out means as separate sensitivity.

The eight-residual-df and two-thirds-positive-fold rules are conventions, not universal constants.

## Pseudocode

```text
INPUT panel data + frozen configuration

0 validate unique unit-time rows, time order, dimensions, missingness
  record seed, software, configuration and data hashes

1 enumerate each (outcome,predictor,direction) and assign testing family

2 diagnostic gate: missingness, persistence/stationarity, common shocks, breaks
  if lag/model design leaves insufficient information: mark not testable

3 association screen
  compute effect-size descriptions
  create circular-shift surrogates within units
  compute surrogate p-values
  apply BY q=.05 primary and BH q=.05 sensitivity

4 classical linear precedence
  for p=1,...,p_max fit restricted and unrestricted models on identical rows
  store all robust nested/Wald p-values
  p_lag = min(1, number_valid_lags * min(raw lag p-values))
  after all edges: q_BY=BY(p_lag), q_BH=BH(p_lag)

5 Bayesian linear comparison
  compare identical restricted/unrestricted information sets at each lag
  portable implementation: log(BF10)=0.5*(BIC_restricted-BIC_unrestricted)
  store maximum log(BF10); support if log(BF10)>=log(5)

6 RF predictive branch
  build forward folds
  fit restricted RF and unrestricted RF adding past x
  delta_f=100*(MSE_R-MSE_U)/MSE_R
  for b=1,...,B_shadow:
      circularly shift x by nonzero amount within unit
      rebuild lags and repeat forward comparison
  Q95=95th percentile shadow gains
  p_shadow=(1+#shadow_gain>=real_gain)/(B_shadow+1)
  support if real_gain>0 AND real_gain>Q95 AND positive_share>=threshold
  repeat across declared seeds/tuning settings for sensitivity

7 placebo gate
  transform full predictor tensor according to declared placebo rule
  compare observed discovery count with placebo distribution

8 common-shock sensitivity
  rerun supported linear evidence with alternative declared control
  preserve disagreements

9 replication gate
  repeat across units or independent layers

10 evidence-family classification
   linear_family = classical OR Bayesian support
   predictive_family = RF support
   if both families + replication + placebo + sensitivity:
       triangulated temporal precedence
   else if predictive only and not replicated:
       nonlinear candidate; replication required
   else if any family + replication:
       replicated single-family evidence
   else if any family:
       exploratory evidence
   else:
       unsupported

11 export complete edge audit and reviewer/reporting summaries
```

## Stopping rules

Do not search indefinitely for favorable lags, seeds, priors, controls, or tuning values. Candidate settings must be frozen before the confirmatory run. Sensitivity variants are reported side by side and do not replace the primary result after inspection. Checkpointed R1 validation resumes only under a matching configuration hash.

## Function signatures

```r
panel_correlation_screen(data, id, time, variables, method, n_surrogates, seed)
classical_panel_precedence(data, id, time, outcome, predictor, max_lag,
                           unit_effects, time_effects, cross_sectional_means, vcov)
bayesian_panel_precedence(data, id, time, outcome, predictor, max_lag,
                          unit_effects, time_effects, cross_sectional_means,
                          bf_threshold)
rf_panel_precedence(data, id, time, outcome, predictor, max_lag, trees,
                    initial_window, horizon, shadow_repeats,
                    positive_fold_share, cross_sectional_means,
                    time_trend, seed)
adjust_families(table, p_col, family_cols, method=c("BY","BH"))
triangulate_evidence(classical_supported, bayesian_supported, rf_supported,
                     replicated_units, replicated_layers,
                     placebo_pass, sensitivity_pass)
```

## Computational scaling

Let E be tested edges, P the maximum lag, F forward folds, B_RF shadows, B_P placebos, and S RF seed/tuning repeats. If C_L is one linear restricted/full fit cost and C_RF one RF fold cost, dominant costs are approximately:

```text
classical: O(E * P * C_L)
portable Bayesian: O(E * P * C_L)
RF: O(E * S * F * (1+B_RF) * C_RF)
linear placebo count: O(B_P * E * P * C_L)
multiplicity: O(E log E) per family
```

Memory is approximately O(N*T*K) plus stored edge results. The RF/shadow branch usually dominates runtime. This motivates the validated 30-tree/19-shadow Monte Carlo approximation while retaining 500/20 as the user-facing reference setting.
