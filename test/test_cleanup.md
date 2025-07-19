# Hammerhead.jl Test Suite Improvement Plan

Prioritized list (highest impact first). Each item: **Problem / Risk → Recommendation**.

---

## 1. Physics / Algorithm Correctness Gaps (Highest Risk)

1. **Unrealistic synthetic image content**  
   *Problem:* Single isolated Gaussian particle; lacks realistic seeding, overlap, noise, illumination gradients.  
   *Recommendation:* Add particle field generator (Poisson-distributed centers, slight diameter/intensity variation, additive Gaussian + Poisson noise, optional background gradient). Test multiple seeding densities (e.g. 0.02, 0.05, 0.1 particles/pixel). Validate RMS displacement error, peak ratio distributions, failure rate.

2. **No tests for large / wrap-around displacements**  
   *Problem:* Displacements never approach ±(window/2); aliasing / periodic ambiguity untested.  
   *Recommendation:* Add cases just below and above the theoretical maximum unambiguous shift; assert correct peak selection or documented failure.

3. **Missing multi-stage & deformation iteration validation**  
   *Problem:* Only single-stage flows; deformation iterations’ benefit unverified.  
   *Recommendation:* Use synthetic flow (uniform translation + shear). Run single vs multi-stage (coarse→fine) with/without deformation; assert error reduction after each stage.

4. **Over-clean subpixel peak tests**  
   *Problem:* Perfect Gaussian at integer location; does not stress asymmetry, noise, saturation, near-equal neighboring peaks.  
   *Recommendation:* Random 3×3 perturbations, flat-top (clipped) peak, closely spaced secondary peak within 1 px, low SNR case. Assert bounded bias & stability (no NaNs).

5. **Quality metric realism**  
   *Problem:* Secondary peak tests use sparse discrete maxima, not continuous correlation structure.  
   *Recommendation:* Generate realistic correlation planes; inject controlled twin peaks; compare robust vs non-robust methods for close / distant secondaries.

6. **Affine transform validation semantics**  
   *Problem:* Reflections accepted; anisotropic stretch with det=1 passes without shape constraint.  
   *Recommendation:* Decide policy: require det>0? Bound condition number / aspect ratio (σ_max/σ_min). Add corresponding tests.

---

## 2. Determinism & Reproducibility

1. **Inconsistent RNG seeding**  
   *Recommendation:* Seed every testset (e.g. helper `@withseed`). Avoid global state leakage.

2. **Noisy performance test**  
   *Recommendation:* Warm-up outside timing; optionally gate perf tests with `ENV["RUN_PERF_TESTS"]`; consider `BenchmarkTools.@belapsed`.

---

## 3. Edge Case & Negative Input Coverage

1. **NaNs / Inf in input images**  
   *Recommendation:* Tests with masked regions (NaNs) & infinities: expect graceful handling (ignore, replace) or explicit error.

2. **Uncovered image types**  
   *Recommendation:* Add UInt8 / Gray{N0f8} input tests (convert or reject); ensure complex inputs rejected.

3. **Non-square windows & anisotropic overlap in full pipeline**  
   *Recommendation:* Correlate using (e.g.) (48,32) windows, anisotropic overlap & non-uniform displacement field.

4. **Border + padding interaction**  
   *Recommendation:* Particle partially outside image; ensure padding logic & coordinate reporting unaffected.

---

## 4. Allocation & Type Stability

1. **Single allocation check**  
   *Recommendation:* Post-warm loop verifying near-zero allocations for Float32 & Float64 across multiple runs.

2. **Lack of type inference assertions**  
   *Recommendation:* Use `@inferred` on `correlate!`, `analyze_correlation_plane`, interpolation functions.

---

## 5. API Contract & Invariants

1. **Source image immutability not verified**  
   *Recommendation:* Hash / checksum before & after correlation; assert unchanged.

2. **Degenerate correlation plane behavior undefined**  
   *Recommendation:* All-zero / uniform plane test; document & assert returned displacement (e.g. (0,0) + flagged quality metrics or NaNs).

3. **Interpolation invariants**  
   *Recommendation:* Weights sum to 1 inside hull; linearity: `f(values*α+β) == f(values)*α+β`; outside-hull fallback semantics.

---

## 6. Numerical Robustness & Tolerances

1. **Overly tight absolute tolerances (`1e-10`)**  
   *Recommendation:* Use relative or physics-based tolerance (e.g. `atol = 0.02σ` or CRLB estimate).

2. **Brittle window center value checks**  
   *Recommendation:* Replace magic center constants with invariants (symmetry, monotonic rise to center, edge ~0, normalized max ≈1 within tolerance).

---

## 7. Test Code Issues / Smells

1. **Duplicate Gaussian exponent computation**  
   *Recommendation:* Reuse computed `v`; correct/comment true relationship between “diameter” & σ (e.g. define FWHM or 2σ consistently).

2. **(i,j) vs (x,y) naming inconsistency**  
   *Recommendation:* Standardize coordinate semantics; adjust variable names & comments.

3. **Repeated nearest-vector selection logic**  
   *Recommendation:* Extract helper function.

4. **Monolithic test file**  
   *Recommendation:* Split into thematic files (`synthetic`, `correlator`, `subpixel`, `quality`, `stages`, `windows`, `interpolation`, `affine`, `integration`, `perf`).

---

## 8. Coverage Gaps

- Failure branches (e.g. secondary peak exclusion eliminating all candidates).
- Metadata merging (if/when implemented).
- Future GPU / multithread paths (placeholder conditional tests).
- Use Coverage.jl; track uncovered lines in CI.

---

## 9. Property-Based / Fuzz Testing

- Random displacement tests (uniform within ±5 px) → assert mean abs error < threshold.
- Random affine transforms within tolerance accepted; slightly exceeding rejected.
- Random triangles & barycentric weight recovery.

---

## 10. Performance & Scaling

1. **Limited size range**  
   *Recommendation:* Optional scaling tests at 256², 512² (env-gated); assert near O(N log N) timing ratio.

2. **Resizing correlator**  
   *Recommendation:* If resizing supported, test plan reallocation / reuse; if not, assert clear error on mismatch.

---

## 11. Doctest Integration

- Convert exemplar tests (simple correlation, multi-stage pipeline) to doctests embedded in docstrings.

---

## 12. Interpolation & Status Handling Semantics

- Multiple adjacent `:bad` vectors (hole larger than 1 cell).
- Mixed source statuses precedence (e.g. prefer `:good` over `:secondary`).
- Confirm original `PIVResult` immutability or document in-place mutation.

---

## 13. Window Function Parameter Sweeps

- Tukey α in {0.1,0.5,0.9}; Kaiser β in {2,5,8}; Gaussian σ fraction variations.  
- Assert energy ordering and smooth monotonic relationship of peak attenuation.

---

## 14. Assertion Refinement

- Replace exact numeric peak ratios with bounded inequalities (e.g. `1 < peak_ratio < 1e6`).
- Use expressive custom predicate helpers (`@test is_valid_peak_ratio(r)`).

---

## 15. Test Utilities (DRY)

Introduce reusable helpers:
- `generate_particle_field(size; N, diam_mean, diam_std, noise, gradient)`
- `nearest_vector(result, x, y)`
- `with_seed(id) do ... end`
- `assert_displacement(du,dv,true_u,true_v; tol)`

---

## 16. CI Enhancements

- Matrix: Julia versions, `JULIA_NUM_THREADS=1,4`.
- Fast vs extended test sets (ENV flags).
- Performance baseline artifact (JSON) for trend monitoring.
- Fail on coverage regression (threshold drop).

---

## 17. Guard Rails for Future GPU / Parallel Features

- Conditional test comparing CPU vs GPU correlation (if CUDA available) for one case.
- Thread-safety test running multiple correlations in parallel tasks (if safe).

---

## 18. Additional Numerical Scaling Checks

- Assert energy ordering: `E_rect ≥ E_hamming ≥ E_hann ≥ E_blackman` (or correct expected ordering) for same length.

---

## 19. Gaussian Diameter Definition

- Choose: diameter = FWHM, or diameter = 2σ, etc.  
- Add test verifying measured second moment matches chosen definition within tolerance (e.g. <5% deviation).

---

## 20. Minor Housekeeping

- Explicit `rtol` / `atol` for Float32.
- Shorten overly verbose test names.
- Remove non-essential debug asserts (`distance_to_close`) or move to helper.
- Consistent status vocabulary (`:bad`, `:interpolated`, `:repaired`, etc.) tests for allowed transitions.

---

## Sample Additions (Illustrative)

**Random displacement property test:**
```julia
@testset "Random displacement accuracy" begin
    Random.seed!(123)
    cc = CrossCorrelator((64,64))
    for trial in 1:25
        disp = (rand()*10 - 5, rand()*10 - 5)  # ±5 px
        img1, img2 = synth_field((64,64); particles=800, diameter=2.5, displacement=disp, noise=0.02)
        corr = correlate!(cc, img1, img2)
        du, dv, pr, cm = analyze_correlation_plane(corr)
        @test abs(du - disp[1]) < 0.05
        @test abs(dv - disp[2]) < 0.05
        @test pr > 1
    end
end
```

**Allocation stability (post-warm):**
@testset "Allocation-free correlate!" begin
    Random.seed!(1)
    cc = CrossCorrelator((64,64))
    img1 = rand(Float32,64,64); img2 = rand(Float32,64,64)
    correlate!(cc, img1, img2) # warm
    alloc = @allocated correlate!(cc, img1, img2)
    @test alloc == 0 || alloc < 256
end

## Highest Priority Summary
1. Realistic multi-particle noisy fields & broader displacement cases.
2. Multi-stage / deformation efficacy tests.
3. Robust subpixel & quality metric stress tests.
4. Deterministic seeding & property-based fuzz tests.
5. Expanded edge-case coverage (NaNs, borders, non-square windows).
6. Allocation & type stability assertions.
7. Refactor brittle numeric assertions → invariant-based checks.
8. Modularize tests and add utilities.
