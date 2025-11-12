PCA-sampling in xalgo-sport-tribes — Full description

Generated: November 12, 2025
Location: `/Users/justho/Downloads/pca-description.md`

Goal
- Describe exactly how the project produces parameter samples via the `sampler.pca_sample` call, how wrappers use those samples to produce parameter vectors, the statistical consequences, and recommended changes if you want a true PCA-based or covariance-accurate sampler.

What `sampler.pca_sample` actually does (exact behavior)
- Location of implementation: `dist/export/.../site-packages/xalgo_schemas/calculation_unit/sampling.py`
- `pca_sample(sim_id, rows, cols)` steps:
  1. Build a deterministic PRNG seed from `sim_id` using a custom SeedSequence subclass `__PCG64SeedSequece`. It uses the last 32 hex chars of the SimulationXID as a big integer to initialize the PCG64 RNG deterministically per simulation id.
  2. Call `Sampling.sample_bytes(sim_id, 2 * rows * cols)` to produce `2*rows*cols` bytes via PCG64.
  3. Interpret bytes as little-endian unsigned 16-bit integers: `x = np.frombuffer(b, dtype="<u2")`.
  4. Load a palette `bm16 = Sampling.__palette_16()` from `16_bit_palette.npz` (cached via lru_cache): this palette is `np.concatenate((-palette, palette))`, then sorted. The palette contains 16-bit values chosen by the project (not standard Gaussian quantiles).
  5. Map the 16-bit integers to palette values: `res = np.take(bm16, x.reshape(-1)).reshape((rows, cols))`.
  6. Center columns to zero mean: `res -= res.mean(axis=0)`.
  7. Return `res` (dtype float32) as `standard_sample` with shape (rows, cols).

How wrappers use `pca_sample`
- Call site: `applications/*/*/core_model/wrapper.py` (soccer, basketball, ice-hockey, tennis wrappers).
- Steps in wrapper:
  - `mu = simulation_job.parameter_state.fast` (1D length p)
  - `sigma = sqrt(sigma_shared**2 + sigma_wise**2)` (1D length p)
  - `standard_sample = sampler.pca_sample(sim_id, rows=size, cols=len(mu))`  # zero-mean matrix
  - `sample = (standard_sample * sigma + mu).astype(np.float32)`  # elementwise multiply and add
- The returned `sample` is used across the wrapper to extract parameter columns (e.g., `draw_factor = sample[:, fast_keys.index('draw_factor')]`) which in turn influence the simulation (draw weighting, intensities, etc.).

Statistical interpretation and consequences
- Despite the “pca” name, this method does NOT perform Principal Component Analysis or SVD.
- `standard_sample` is a deterministic column-centered palette-mapped pseudo-random array. Any cross-column dependence comes indirectly from how bytes are consumed and how palette mapping clusters values. There is no guarantee that columns are independent or that they have a given correlation matrix.
- Scaling by `sigma` and adding `mu` sets per-parameter marginal mean and scale, but does not guarantee any particular cross-parameter covariance structure beyond what the palette-based generator already induced.
- Because columns are centered individually, marginal means will be controlled precisely by adding `mu`. Marginal variances are controlled approximately by `sigma` times the palette spread, but exact target variances require the palette and post-scaling to be consistent.

Determinism and reproducibility
- Deterministic: with the same `sim_id` and the same rows/cols, `pca_sample` returns the same matrix.
- Seeding uses `SimulationXID` to produce the seed; different simulation ids produce (effectively) independent streams.

Why this approach might be used
- Speed and portability: avoids heavy linear algebra or external RNG state; palette-based mapping can be fast and deterministic across platforms.
- Compatibility: same deterministic mapping can be reproduced across services and languages if palette and mapping code are shared.
- Simplicity: elementwise scaling and shifting is simple to apply after the base draws are produced.

Limitations and risks (when compared with a true PCA or Cholesky sampler)
- It is not a mathematically guaranteed multivariate Gaussian generator with a specified covariance. If your goal is to sample from a particular multivariate normal (μ, Σ), this method is inappropriate unless the palette and byte mapping were designed to approximate that Σ (which is unlikely without explicit design).
- Interpreting `pca_sample` as PCA-based will be misleading for future readers and contributors.
- Tail behavior and dependence structure: palette mapping may not provide Gaussian tails (it is discrete and bounded by palette values), affecting any heavy-tail-sensitive metrics.

Suggested fixes / alternatives (if you want true PCA sampling)
1. Replace or wrap `pca_sample` with a sampler that draws exact multivariate Gaussian samples for target covariance Σ: use Cholesky or SVD-based transform.
   - If Σ = diag(sigma)**2 (independent per-dimension) then simple `np.random.normal(size=(rows, cols)) * sigma[None, :] + mu[None, :]` suffices.
   - If Σ is full covariance and you want to sample from N(mu, Σ): precompute A with `A = U @ sqrt(Λ)` from eigen-decomp or `L` from cholesky, then do `samples = np.random.normal(size=(rows, k)) @ A.T + mu` (with optional truncation/residual handling).
2. If correlation structure is desired but you want reproducibility per `sim_id`, keep a seeded RNG but implement correct transforms:
   - seed = derived from sim_id
   - rng = np.random.default_rng(seed)
   - standard = rng.standard_normal(size=(rows, cols))
   - if covariance desired: standard @ A.T + mu
3. If you want a fast, deterministic, repeatable generator, but with correct marginal/covariance properties, you can keep a palette approach but design the palette and indexing so the resulting correlational structure matches target Cov — this is complex and brittle; prefer mathematically-correct transforms instead.

Quick code example (exact multivariate Gaussian sampling with seed derived from sim_id):
```python
from numpy.random import default_rng

seed = some_hash(sim_id)
rg = default_rng(seed)
Z = rg.normal(size=(rows, p))  # independent standard normals
# precompute A such that Sigma = A @ A.T (e.g., via eigh or cholesky)
X = Z @ A.T + mu[None, :]
```

Actionable next steps I can do for you
- Produce a `pca-description.md` with this summary (done and saved to your Downloads).
- Run a small script that compares `Sampling.pca_sample` empirical covariance to diag(sigma)**2 and/or to a target covariance for a small p (I can run it locally here to produce numbers/plots).
- Propose and optionally patch a replacement `pca_sample` that uses seeded GN and either Cholesky or PCA for exact covariance sampling (with feature toggle so existing deterministic behavior remains available).

Status
- I created `/Users/justho/Downloads/pca-description.md` with the content above.

Which of the follow-ups should I do next? I can (A) run a quick empirical test showing the palette-based sampler's empirical covariance and marginals, (B) implement a seeded true-Gaussian sampler and test equivalence/differences, or (C) create a documented wrapper that exposes both behaviors and can be toggled via config. Which would you like? 