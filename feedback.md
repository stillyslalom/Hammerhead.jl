# feedback.md — remaining items

The other seven items from the original review were implemented on branch
`feedback-batch-1` (see `FEEDBACK_PLAN.md` Part A). These two were reserved
because they touch the correlation hot path and accuracy-tolerance
conventions — they need benchmarking/design iteration, not just execution.

* **DONE (branch `feedback-b1`, see FEEDBACK_PLAN.md B1: `max_iterations` +
  `convergence_tol`; per-vector tracking rejected on benchmark evidence).**
  Multipass should be *iterative* per-stage. If a vector in the first stage fails, the failure can cascade to later stages. Better to iteratively perform validation, bad-vector replacement, smoothing, and deformed correlation until a given stage converges (or maxiters reached). It may be worthwhile to track per-vector convergence; converged vectors surrounded by other converged vectors probably don't need to be re-correlated on the nth iteration. Similarly, a vector with terrible correlation statistics surrounded by *other* terrible vectors probably doesn't stand to gain much from the nth iteration.
* The peak-finding routine currently uses an exclusion radius to exclude larger peaks from secondary/tertiary peak search. This would be problematic for true secondary peaks which lie within the exclusion radius, as might happen with a strong primary peak produced by stationary lab equipment and a nearby secondary peak from small-displacement flow. Prana (Matlab PIV toolkit) uses `imregionalmax` to robustly identify local maxima; it's an expensive operation on par with the cost of correlation itself, but would be worth including as an option for gold-plated peakfinding. As an alternative, the shape of the correlation peak and the measured noise floor could be used to dynamically calculate a per-peak exclusion radius.
