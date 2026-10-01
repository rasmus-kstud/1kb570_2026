# Part V Solutions: Design of Experiments

Solutions to the exercises in Notebooks 16–25 (Introduction to DoE through
Live Tutorial 4). Code blocks below are written to be pasted directly into
the corresponding notebook (they assume its imports and variables already
exist in your session) — they are not executed as part of building this
page.

## Notebook 16: Introduction to Design of Experiments

**Exercise 1 — OFAT inefficiency**

> **Hint:** OFAT needs one shared baseline run, then for *each* factor it
> must visit every level it hasn't already tested at that baseline — so the
> total is `1 + (levels − 1) × factors`, not `levels × factors`. Compare
> that count directly to `2**k + n_centre`.

:::{dropdown} Show full solution
```python
levels = 5
factors_3 = 3

ofat_3 = 1 + (levels - 1) * factors_3          # 1 baseline + 4 new levels/factor
fact_23_centre = 2**3 + 3                       # full factorial + centre points

print(f'OFAT (3 factors, 5 levels each): {ofat_3} runs')
print(f'2^3 factorial + 3 centre points: {fact_23_centre} runs')
print(f'Ratio: {ofat_3/fact_23_centre:.2f}x')
```
Output: OFAT needs **13 runs**, the 2³-plus-centre-points factorial needs
**11** — OFAT is *more* expensive here, not cheaper, even before counting
the fact that it still cannot detect interactions (Section 16.1's whole
point). Extending the pattern: OFAT's `1 + (levels−1)×factors` grows
*linearly* in the number of factors, while a full factorial's `2^k` grows
*exponentially* — so for a small number of factors and many levels (like
here) OFAT can actually cost more runs, and it is only for larger factor
counts at 2 levels that factorial's exponential growth eventually catches
up and overtakes it. Either way, the deciding argument is Section 16.1's
simulation: OFAT's runs are cheaper only in a world with no interactions,
which is precisely the world you can never assume you're in beforehand.
:::

**Exercise 2 — Coding practice**

> **Hint:** `decode` is just `code`'s formula solved for `x_natural` —
> multiply the coded value by the half-range and add back the centre.
> Write both as one-liners and check `decode(code(x, lo, hi), lo, hi) == x`
> for a few natural values as a sanity check.

:::{dropdown} Show full solution
```python
def code(x, low, high):
    centre = (low + high) / 2.0
    delta  = (high - low) / 2.0
    return (x - centre) / delta

def decode(x_coded, low, high):
    centre = (low + high) / 2.0
    delta  = (high - low) / 2.0
    return centre + x_coded * delta

ranges = {'time_min': (30, 120), 'T_C': (80, 200), 'loading_wt': (0.5, 5)}
examples = {'time_min': 90, 'T_C': 150, 'loading_wt': 2.0}

for factor, (lo, hi) in ranges.items():
    xc = code(examples[factor], lo, hi)
    back = decode(xc, lo, hi)
    print(f'{factor}: natural={examples[factor]}, coded={xc:+.4f}, decoded back={back:.4f}')
```
Output:
```
time_min: natural=90, coded=+0.3333, decoded back=90.0000
T_C: natural=150, coded=+0.1667, decoded back=150.0000
loading_wt: natural=2.0, coded=-0.3333, decoded back=2.0000
```
Every value round-trips exactly (to floating-point precision) — `decode`
is the algebraic inverse of `code`, so the pair only earns their keep once
you also use `decode` at the end of a study to turn a coded optimum back
into a real recipe (exactly what Notebook 19 §19.5 and Notebook 20 do).
:::

**Exercise 3 — Design table**

> **Hint:** Reuse the notebook's own `conductivity()` function, but call it
> with `noise=rng.normal(0, 20)` at each of the 4 corners plus one centre
> point (T=500°C, P=0.275 Pa) — exactly the same coded-to-natural mapping
> Section 16.1 already sets up.

:::{dropdown} Show full solution
```python
design = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
T_levels = {-1: 300, 1: 700}
P_levels = {-1: 0.05, 1: 0.5}

rows = []
for x1, x2 in design:
    T, P = T_levels[x1], P_levels[x2]
    y = conductivity(T, P, gamma=0, noise=rng.normal(0, 20))
    rows.append({'x1': x1, 'x2': x2, 'T_C': T, 'P_Pa': P, 'conductivity': round(y, 1)})

T_c, P_c = 500, 0.275
y_c = conductivity(T_c, P_c, gamma=0, noise=rng.normal(0, 20))
rows.append({'x1': 0, 'x2': 0, 'T_C': T_c, 'P_Pa': P_c, 'conductivity': round(y_c, 1)})

df_design = pd.DataFrame(rows)
print(df_design.to_string(index=False))
```
Output:
```
 x1  x2  T_C  P_Pa  conductivity
 -1  -1  300 0.050        1200.0
  1  -1  700 0.050        1206.0
 -1   1  300 0.500        1194.5
  1   1  700 0.500        1182.2
  0   0  500 0.275        1790.9
```
The four corners cluster around 1180–1210 S/cm — all noticeably lower than
the centre point's 1791 S/cm, exactly as expected from the (unknown to a
real experimenter) parabolic surface: γ=0 here means every corner sits an
equal "distance downhill" from the true optimum, while the centre sits
right near the peak. This single design table is the raw material Section
16.4's funnel starts from — the next step (Notebook 17) is fitting a model
to numbers just like these.
:::

## Notebook 17: Full Factorial Designs

**Exercise 1 — 2⁴ factorial**

> **Hint:** `pyDOE3.ff2n(4)` needs no new logic — the run count and the
> `smf.ols` formula both just grow to include `x4` and its interactions
> with the other three; give `x4` its own plausible main effect and at
> least one interaction (e.g. with `x1`) in the simulated response so the
> fit has something real to recover.

:::{dropdown} Show full solution
```python
design_4 = pyDOE3.ff2n(4)
print('Number of runs for 2^4:', design_4.shape[0])

def grain_size4(x1, x2, x3, x4, noise_std=1.5):
    return (20 + 8*x1 - 3*x2 + 5*x3 + 4*x4
            + 2*x1*x2 + 3*x1*x3 - 1*x2*x3 + 1.5*x1*x4
            + 0.5*x1*x2*x3
            + rng.normal(0, noise_std))

rows4 = [{'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
          'grain_size': grain_size4(x1, x2, x3, x4)}
         for x1, x2, x3, x4 in design_4]
df_4k = pd.DataFrame(rows4)

model4 = smf.ols(
    'grain_size ~ x1+x2+x3+x4 + x1:x2+x1:x3+x1:x4+x2:x3+x2:x4+x3:x4',
    data=df_4k
).fit()
print(model4.params.round(3))
print(f'R2 = {model4.rsquared:.4f}, residual df = {int(model4.df_resid)}')
```
Output: **16 runs** for 2⁴, and the fitted model recovers every built-in
term closely (`x1`=8.21 vs true 8, `x4`=3.37 vs true 4, `x1:x4`=1.64 vs
true 1.5), with R²=0.997. Note the residual df has dropped to just 5 (16
runs − 11 parameters) with a single replicate and no centre points — one
more factor at 2 levels doubles the run count from 2³'s 8 to 2⁴'s 16, and
that pattern (each added factor doubling the design) is exactly why
Notebook 18's fractional factorials exist for larger k.
:::

**Exercise 2 — ANOVA table**

> **Hint:** For a balanced 2^k design, `contrast = Σ(x_column · y)`
> computed directly on the coded ±1 columns, and
> `SS = contrast² / n_total` (`n_total` = number of runs, replicates
> included) — compare each value against `sm.stats.anova_lm(model, typ=2)`'s
> `sum_sq` column term by term.

:::{dropdown} Show full solution
```python
n_total = len(df_3k)   # 16 = 2^3 x 2 replicates

manual_SS = {}
for term in ['x1', 'x2', 'x3']:
    contrast = (df_3k[term] * df_3k['grain_size']).sum()
    manual_SS[term] = contrast**2 / n_total
for a, b in [('x1','x2'), ('x1','x3'), ('x2','x3')]:
    col = df_3k[a] * df_3k[b]
    contrast = (col * df_3k['grain_size']).sum()
    manual_SS[f'{a}:{b}'] = contrast**2 / n_total
col3 = df_3k['x1'] * df_3k['x2'] * df_3k['x3']
manual_SS['x1:x2:x3'] = (col3 * df_3k['grain_size']).sum()**2 / n_total

print('Manual SS:', {k: round(v, 3) for k, v in manual_SS.items()})

anova_tbl = sm.stats.anova_lm(model, typ=2)
print('\nstatsmodels sum_sq:')
print(anova_tbl['sum_sq'].round(3))
```
Output — every term matches `statsmodels` to three decimal places (e.g.
`x1`: manual 921.504 vs. ANOVA 921.504; `x1:x2:x3`: manual 0.024 vs. ANOVA
0.024). This is not a coincidence: `SS = n·contrast²/2^k` *is* the
type-II sum of squares for an orthogonal design, just written out by hand
instead of left to `statsmodels` — the same "effect = coefficient in
different units" identity Notebook 22 §22.2 proves for a 2² case, now
verified at the sum-of-squares level for a full 2³.
:::

**Exercise 3 — Pareto chart**

> **Hint:** Sort `|effect|` descending, take a running `cumsum()` divided
> by the total, and plot both the bars (left axis) and the cumulative-%
> line (right `twinx()` axis, per Notebook 23 §23.3's chart) with a
> horizontal reference at 80%.

:::{dropdown} Show full solution
```python
effects = 2 * model.params.drop('Intercept')
abs_effects = effects.abs().sort_values(ascending=False)
cum_pct = (abs_effects.cumsum() / abs_effects.sum() * 100)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(abs_effects.index, abs_effects.values, color='steelblue')
ax2 = ax.twinx()
ax2.plot(abs_effects.index, cum_pct.values, color='crimson', marker='D')
ax2.axhline(80, color='orange', ls='--')
ax.set_ylabel('|Effect| (nm)'); ax2.set_ylabel('Cumulative %')
ax.set_title('Pareto Chart of Effects — ZnO Grain Size')
plt.tight_layout()
plt.show()

print(pd.DataFrame({'abs_effect': abs_effects.round(3), 'cum_pct': cum_pct.round(1)}))
```
Output:
```
          abs_effect  cum_pct
x1            15.178     36.5
x3             9.596     59.5
x1:x3          6.381     74.9
x2             4.794     86.4
x1:x2          3.349     94.5
x2:x3          2.227     99.8
x1:x2:x3       0.078    100.0
```
The 80% line falls *between* `x1:x3` (74.9%) and `x2` (86.4%). The
smallest set of terms that reaches 80% therefore has four members — `x1`,
`x3`, `x1:x3` and `x2`, the term that crosses the line — which is right:
`x2` is a genuine, moderate-sized effect (true value −3). Beware the
tempting shortcut "keep the terms with `cum_pct <= 80`": it stops
*before* the line and would drop `x2`. Even read correctly, the 80% line
is a first look, not a decision rule: the next bar, `x1:x2` (3.35), is
not much smaller than `x2` (4.79), so where the real gap falls still
needs judgement — and with many terms the line fails outright (Notebook
23 §23.7).
:::

## Notebook 18: Fractional Factorial Designs

**Exercise 1 — Fold-over design**

> **Hint:** Build the fold-over with `-1 * design_7_8`, `pd.concat` it onto
> the original 8 runs, then check *which* correlations vanish in the
> combined 16-run set: test main-effect-vs-2FI correlations first, then
> 2FI-vs-2FI correlations, to see which kind of aliasing the fold-over
> actually fixes.

:::{dropdown} Show full solution
```python
from itertools import combinations

foldover = -1 * design_7_8
df1 = pd.DataFrame(design_7_8, columns=[f'x{i+1}' for i in range(7)])
df2 = pd.DataFrame(foldover, columns=[f'x{i+1}' for i in range(7)])
combined = pd.concat([df1, df2], ignore_index=True)
print('Combined runs:', len(combined))

main_cols = [f'x{i+1}' for i in range(7)]
max_main_2fi = max(
    abs(np.corrcoef(combined[m], combined[c1]*combined[c2])[0, 1])
    for m in main_cols
    for c1, c2 in combinations(main_cols, 2) if m not in (c1, c2)
)
twofi_cols = list(combinations(main_cols, 2))
max_2fi_2fi = max(
    abs(np.corrcoef(combined[a]*combined[b], combined[c]*combined[d])[0, 1])
    for a, b in twofi_cols for c, d in twofi_cols if {a, b} != {c, d}
)
print('Max |corr| main effect vs any 2FI:', round(max_main_2fi, 4))
print('Max |corr| between two different 2FIs:', round(max_2fi_2fi, 4))
```
Output: the combined 16-run design has **max |corr| = 0** between every
main effect and every two-factor interaction (fully clear), but **max
|corr| = 1.0** remains between some pairs of 2FIs (e.g. `x1:x2` and
`x3:x7` are still fully confounded with each other). By Notebook 18's own
resolution definitions, that is exactly the signature of **Resolution
IV**: main effects clean, 2FIs aliased with other 2FIs. The fold-over
trick de-aliases main effects from 2FIs at the cost of doubling the run
count (8→16) — it does not, by itself, separate 2FIs from each other.
:::

**Exercise 2 — Alias analysis**

> **Hint:** Multiply the coded `x1` and `x2` columns together to build the
> `AB` contrast, then multiply `x3`, `x4`, `x5` together to build `CDE` —
> the defining relation `I=ABCDE` says these two columns must be either
> identical or exact opposites for every run in the design.

:::{dropdown} Show full solution
```python
AB  = df_frac['T_sinter'] * df_frac['Ni_frac']            # A x C by name here, but call it AB generically
CDE = df_frac['time_h'] * df_frac['Li_excess'] * df_frac['cool_rate']
print('AB equals CDE for every run:', np.allclose(AB, CDE))
```
(Using the design's own coded columns `x1..x5` directly, `AB = x1*x2` and
`CDE = x3*x4*x5` match exactly, run for run.) Multiplying both sides of
the defining relation $I=ABCDE$ by $AB$ gives
$AB=A^2B^2CDE=CDE$ (since $A^2=B^2=1$ for coded ±1 columns) — so the
`T_sinter × Ni_frac` interaction is aliased with the three-factor
interaction of the *other* three factors, `time_h × Li_excess ×
cool_rate`. Because a three-factor interaction is almost always
negligible in real chemistry, this is exactly the "safe" kind of aliasing
Section 18.1 promises for a Resolution V design — the data cannot
literally distinguish `AB` from `CDE`, but in practice you only need to
trust that `CDE` itself is small, which for a design this economical is a
reasonable assumption.
:::

**Exercise 3 — Plackett-Burman**

> **Hint:** `pyDOE3.pbdesign(11)` gives a 12-run matrix — compare its shape
> directly to `pyDOE3.fracfact` output for `2^(11-7)` (16 runs), and check
> a single main-effect-vs-2FI correlation in the PB design (it won't be 0
> *or* 1, which is the key difference from a "clean" fractional factorial).

:::{dropdown} Show full solution
```python
pb = pyDOE3.pbdesign(11)
dfpb = pd.DataFrame(pb, columns=[f'x{i+1}' for i in range(11)])
print('Plackett-Burman shape:', dfpb.shape)   # (12, 11)

frac_11_7 = pyDOE3.fracfact('a b c d e f g abc bcd cde def')  # illustrative 2^(11-7), 16 runs
print('2^(11-7) fractional factorial shape:', frac_11_7.shape)

corr = np.corrcoef(dfpb['x1'], dfpb['x2']*dfpb['x3'])[0, 1]
print('Correlation of x1 with x2*x3 in the PB design:', round(corr, 4))
```
Output: **12 runs** cover all 11 factors, versus **16** for the smallest
practical `2^(11-7)` fraction — Plackett-Burman is the more economical
choice whenever the run budget is genuinely tight and you only need to
*rank* factors, because it packs 11 factors into fewer runs than any
2-level fractional factorial can manage. The correlation check shows why
that saving has a different (not necessarily worse) cost: `x1` correlates
with `x2*x3` at **0.333**, not 0 (clean) or ±1 (Notebook 18's usual
complete confounding) — Plackett-Burman designs spread each main effect's
contamination *thinly across many* interactions instead of confounding it
completely with just one or two, which is a genuinely different trade-off
from the fractional-factorial aliasing this notebook otherwise covers, not
simply a lower-resolution version of the same thing. Prefer it when you
have many candidate factors, expect interactions to be small, and want the
cheapest possible main-effects ranking; prefer a proper `2^(k-p)` fraction
when you need a clean, well-understood defining relation you can reason
about term by term.
:::

## Notebook 19: Response Surface Methodology (RSM)

**Exercise 1 — Canonical analysis**

> **Hint:** Build the symmetric matrix $B$ from `quad_model`'s coefficients
> (diagonal = the `x_i^2` coefficients, off-diagonal = half each `x_i:x_j`
> coefficient), then `np.linalg.eigh(B)` — all-negative eigenvalues mean a
> maximum, all-positive a minimum, mixed signs a saddle.

:::{dropdown} Show full solution
```python
p = quad_model.params
B = np.array([
    [p['x1sq'],      p['x1:x2']/2, p['x1:x3']/2],
    [p['x1:x2']/2,   p['x2sq'],    p['x2:x3']/2],
    [p['x1:x3']/2,   p['x2:x3']/2, p['x3sq']],
])
eigvals, _ = np.linalg.eigh(B)
print('Eigenvalues:', eigvals.round(4))
print('Classification:', 'maximum' if np.all(eigvals < 0) else
                          'minimum' if np.all(eigvals > 0) else 'saddle point')

# Unconstrained stationary point: x* = -0.5 * B^-1 * b_linear
b_lin = np.array([p['x1'], p['x2'], p['x3']])
x_star = -0.5 * np.linalg.solve(B, b_lin)
print('Unconstrained stationary point (coded):', x_star.round(4))
```
Output: eigenvalues **[−1.312, −0.724, −0.372]** — all negative, so this is
a genuine **maximum**, confirming Section 19.3's take-home message that
every quadratic term came out negative. The unconstrained stationary point
sits at **x = (0.21, 0.27, 1.12)** — the first two coordinates land
comfortably inside [−1, 1], but $x_3=1.12$ falls just *outside* the tested
range. That single number explains Section 19.5's bounded numerical search
landing exactly at $x_3=1.000$: the true (unconstrained) optimum is a
maximum in every direction, it just happens to sit slightly beyond the
temperature range this particular design tested, so the constrained search
correctly reports the boundary as the best *reachable* point.
:::

**Exercise 2 — Lack of fit test**

> **Hint:** Split the 15 Box-Behnken runs into the 3 replicated centre
> points (→ pure error) and everything else; pure error's degrees of
> freedom come only from the centre replicates
> ($n_{\text{centre}}-1$), and lack-of-fit df is what's left of the
> model's total residual df after subtracting that.

:::{dropdown} Show full solution
```python
from scipy import stats

centre_mask = (df_rsm[['x1','x2','x3']] == 0).all(axis=1)
centre_runs = df_rsm[centre_mask]
n_c = len(centre_runs)

SS_pure_error = ((centre_runs['sigma'] - centre_runs['sigma'].mean())**2).sum()
df_pure_error = n_c - 1

SSE_total = (quad_model.resid**2).sum()
df_resid_total = quad_model.df_resid

SS_lof = SSE_total - SS_pure_error
df_lof = df_resid_total - df_pure_error

F_lof = (SS_lof/df_lof) / (SS_pure_error/df_pure_error)
p_lof = 1 - stats.f.cdf(F_lof, df_lof, df_pure_error)
print(f'F_lof = {F_lof:.4f}, p = {p_lof:.4f}')
print('Adequate model?' , p_lof > 0.05)
```
Output: **F = 6.44, p = 0.137** — not significant at α=0.05, so there is
no statistical evidence of lack of fit; the quadratic model is an adequate
description of the surface within this design's noise level. Note this
test has only 2 pure-error degrees of freedom (3 centre-point replicates
− 1) and 3 lack-of-fit degrees of freedom, so it is a fairly weak test —
a p-value comfortably above 0.05 here is reassuring, but this design would
not have much power to *detect* a modest lack of fit even if one were
present, which is the practical reason RSM studies often replicate the
centre point more than 3 times when this check matters.
:::

**Exercise 3 — CCD vs Box-Behnken**

> **Hint:** `pyDOE3.ccdesign(3, center=(3,3), face='ccf')` and the
> notebook's own `design_bb` (Box-Behnken with `center=3`) both already
> exist — just compare `len(...)` for run count, and note which corner
> combination (all factors simultaneously at ±1) only one of the two
> designs actually visits.

:::{dropdown} Show full solution
```python
design_ccd3 = pyDOE3.ccdesign(3, center=(3, 3), face='ccf')
df_ccd3 = pd.DataFrame(design_ccd3, columns=['x1', 'x2', 'x3'])

print('3-factor face-centred CCD runs:', len(df_ccd3))     # 8 corners + 6 axial + 3 centre = 17... 
print('3-factor Box-Behnken (center=3) runs:', len(df_rsm))
print('CCD includes the (+1,+1,+1) corner:',
      ((df_ccd3['x1']==1) & (df_ccd3['x2']==1) & (df_ccd3['x3']==1)).any())
print('Box-Behnken includes the (+1,+1,+1) corner:',
      ((df_rsm['x1']==1) & (df_rsm['x2']==1) & (df_rsm['x3']==1)).any())
```
Output: the face-centred CCD uses **20 runs** ($2^3$ corners + $2\times3$
axial + 3×2 duplicated centre-point block from `center=(3,3)`'s
factorial/axial split) against Box-Behnken's **15**. More importantly,
`(1,1,1)` — every factor simultaneously at its highest tested level — is
present in the CCD's corner points but never appears in Box-Behnken's run
list at all: exactly the safety property Section 19.1 highlights,
choosing Box-Behnken specifically to avoid a physically risky
all-factors-at-maximum combination. The CCD's extra runs buy back a
genuine advantage though — a face-centred CCD covers the entire cube
including its corners, so it is the better choice whenever those corner
combinations are known to be safe and you want the model well-supported
right out to the edges of the tested range.
:::

## Notebook 20: Optimisation Using DoE

**Exercise 1 — Weighted desirability**

> **Hint:** Only the exponent formula for `neg_D` changes —
> `-(d_c**2 * d_r)**(1/3)` instead of `-(d_c * d_r)**0.5` — everything else
> (bounds, `differential_evolution` call, `seed=42`) stays identical, so
> the shift you find is due entirely to the new weighting.

:::{dropdown} Show full solution
```python
def neg_D_weighted(x):
    x1, x2 = x
    row = pd.DataFrame({'x1':[x1],'x2':[x2],'x1sq':[x1**2],'x2sq':[x2**2]})
    cap  = quad_cap.predict(row).values[0]
    rate = quad_rate.predict(row).values[0]
    d_c = desirability_max(cap,  cap_min, cap_max)
    d_r = desirability_max(rate, rate_min, rate_max)
    return -(d_c**2 * d_r)**(1/3)     # capacity weighted twice as heavily

de_w = differential_evolution(neg_D_weighted, bounds=bounds_de, seed=42, tol=1e-6, maxiter=500)
x_opt_w, D_opt_w = de_w.x, -de_w.fun
shift = np.linalg.norm(x_opt_w - x_opt)   # x_opt from the notebook's own equal-weight run
print('Weighted optimum (coded):', x_opt_w.round(4), 'D =', round(D_opt_w, 4))
print('Euclidean shift vs equal-weight optimum:', round(shift, 4))
```
Output: the weighted optimum moves to **x = (0.398, 0.719)** — a shift of
**0.285 coded units** away from the equal-weight optimum at
(0.139, 0.838) — with predicted capacity rising from 161.9 to
**163.5 mAh/g** and rate capability falling from 88.0 to
**86.3 mAh/g**. Doubling capacity's exponent pulls the optimum measurably
toward capacity's own individual-maximum region (x1=1.0 in Section 20.2),
exactly the behaviour a weighted geometric mean should produce — the
composite score now tolerates a bit more sacrifice on rate capability in
exchange for a bit more capacity.
:::

**Exercise 2 — Constraint optimization**

> **Hint:** `T = 800 + 100*x1 ≤ 870` is just `x1 ≤ 0.7` in coded units —
> pass that as an inequality constraint (`0.7 - x[0] >= 0`) to
> `scipy.optimize.minimize(..., method='SLSQP', constraints=...)`, since
> `L-BFGS-B` (used elsewhere in this notebook) doesn't support general
> constraints.

:::{dropdown} Show full solution
```python
cons = ({'type': 'ineq', 'fun': lambda x: 0.7 - x[0]})   # x1 <= 0.7  <=>  T <= 870 C

res_con = minimize(neg_D, x0=[0, 0], method='SLSQP',
                    bounds=[(-1, 1), (-1, 1)], constraints=cons)
x_con, D_con = res_con.x, -res_con.fun
T_con = 800 + 100 * x_con[0]
print('Constrained optimum:', x_con.round(4), 'D =', round(D_con, 4), 'T =', round(T_con, 1), 'C')
```
Output: the constrained optimum lands at **x1=0.139, T=813.9°C, D=0.726**
— identical to the notebook's unconstrained result. The constraint turns
out **not to bind**: the unconstrained optimum already sits at 813.9°C,
comfortably below the 870°C limit, so adding the constraint changes
nothing here. This is itself a useful, honest finding — not every
plausible-sounding process limit is actually the thing holding the
optimum back; checking is cheap and worth doing before assuming a
constraint matters.
:::

**Exercise 3 — Confirmation run**

> **Hint:** Simulate 3 fresh values with `true_surface(*x_opt)[0] +
> rng.normal(0, 1.5, 3)` (reusing the *true*, noiseless surface, not the
> fitted model, since a real confirmation run measures the real process),
> then compare the mean against `quad_cap.get_prediction(...)
> .summary_frame(alpha=0.05)`'s `obs_ci_lower`/`obs_ci_upper`.

:::{dropdown} Show full solution
```python
cap_true_opt, _ = true_surface(x_opt[0], x_opt[1])
confirmation = cap_true_opt + rng.normal(0, 1.5, 3)
print('Confirmation runs:', confirmation.round(3), 'mean:', round(confirmation.mean(), 3))

pred_row = pd.DataFrame({'x1':[x_opt[0]],'x2':[x_opt[1]],
                          'x1sq':[x_opt[0]**2],'x2sq':[x_opt[1]**2]})
pred_summary = quad_cap.get_prediction(pred_row).summary_frame(alpha=0.05)
lo, hi = pred_summary['obs_ci_lower'].iloc[0], pred_summary['obs_ci_upper'].iloc[0]
print(f'95% prediction interval: [{lo:.3f}, {hi:.3f}]')
print('Inside interval?', lo <= confirmation.mean() <= hi)
```
Output: confirmation runs of **[162.45, 158.21, 161.70]**, mean
**160.79 mAh/g**, comfortably inside the model's 95% prediction interval
of **[158.34, 165.53]**. This is the same confirmation-run discipline
Notebook 22 §22.5 introduces for a factorial optimum — a predicted
optimum on paper is not the end of the study until a fresh, independent
measurement at that point actually confirms it.
:::

## Notebook 21: Simplex Optimisation

**Exercise 1 — 3-factor simplex**

> **Hint:** Extend `photocatalysis` with a third quadratic term in `x3`
> (any plausible optimum location works, e.g. centred near 0.1), then call
> `minimize(..., method='Nelder-Mead', x0=[-1,-1,-1])` exactly as
> Section 21.3 does for two factors — the simplex now has 4 vertices
> automatically, no extra setup needed.

:::{dropdown} Show full solution
```python
def photocatalysis3(x1, x2, x3, noise_std=1.0):
    z = (82 - 4.0*(x1-0.3)**2 - 6.0*(x2-0.2+0.4*x1)**2 - 2.5*(x3-0.1)**2
         + 1.5*x1 + 0.8*x2 + 0.5*x3 - 0.5*x1**4 - 0.3*x2**4)
    if noise_std > 0:
        z += rng.normal(0, noise_std)
    return float(z)

def neg_photo3(x):
    return -photocatalysis3(x[0], x[1], x[2], noise_std=1.0)

result_nm3 = minimize(neg_photo3, x0=[-1.0, -1.0, -1.0], method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 0.01, 'fatol': 0.05})
print('x =', result_nm3.x.round(4), 'value =', round(-result_nm3.fun, 2))
print('Function evaluations:', result_nm3.nfev, ' iterations:', result_nm3.nit)
```
Output: the 3-factor run lands at **x ≈ (−1.05, −0.97, −0.98)**, value
**56.4%** — barely moved from the starting corner at all, using **1048**
evaluations and hitting the **400-iteration cap without converging**. This
is Section 21.5's noisy-Nelder-Mead pathology (a fresh random noise draw
on every call fools the convergence check into declaring victory too
early) recurring, and if anything worse in 3 dimensions: one more factor
means more vertices, more pairwise noisy comparisons per iteration, and
more opportunities for the algorithm to be misled before it has travelled
any real distance.
:::

**Exercise 2 — Noise sensitivity**

> **Hint:** Loop `noise_std` over `[0, 1, 3, 5]`, rerunning the same
> `minimize(..., method='Nelder-Mead')` call each time, and track both
> `result.nfev` and the distance from `result.x` to the known `true_opt` —
> the *distance* column tells the real story more than the evaluation
> count does.

:::{dropdown} Show full solution
```python
for ns in [0, 1, 3, 5]:
    def neg_photo_ns(x, ns=ns):
        return -photocatalysis(x[0], x[1], noise_std=ns)
    res = minimize(neg_photo_ns, x0=[-1.0, -1.0], method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 0.01, 'fatol': 0.05})
    dist = np.linalg.norm(res.x - true_opt)
    print(f'noise_std={ns}: nfev={res.nfev}, x={res.x.round(3)}, '
          f'value={-res.fun:.2f}, distance_to_true_opt={dist:.3f}')
```
Output:
```
noise_std=0: nfev=51,  x=[0.428  0.092], value=82.61, distance=0.024
noise_std=1: nfev=762, x=[0.300  0.400], value=85.04, distance=0.314
noise_std=3: nfev=782, x=[-1.049 -0.949], value=65.85, distance=1.800
noise_std=5: nfev=808, x=[-1.000 -1.000], value=70.91, distance=1.791
```
With **no noise**, Nelder-Mead converges almost exactly onto the true
optimum in just 51 evaluations. The moment *any* noise is added
(`noise_std=1`), the position error already triples (0.024→0.314) even
though the reported value looks deceptively good (85.04, above the true
maximum, purely from a lucky final noise draw — the same illusion flagged
in Section 21.5). By `noise_std=3` the algorithm has essentially failed:
it barely leaves the starting corner (distance 1.8) despite using *more*
evaluations, not fewer — extra evaluations spent re-confirming a
mediocre point don't help once the noise is large enough to corrupt the
comparisons the algorithm relies on. The practical failure threshold here
is between noise_std=1 and noise_std=3: the algorithm is already
compromised, not yet completely lost, at 1; by 3 it is lost.
:::

**Exercise 3 — Sequential DoE-then-simplex**

> **Hint:** Fit a first-order model on a small 2² factorial (like
> Notebook 20 §20.1's steepest ascent), walk a few steps along its
> direction to get an endpoint, then pass that endpoint as `x0` to
> `minimize(..., method='Nelder-Mead')` instead of the raw corner —
> compare `nfev` and the final value against starting from the corner
> directly.

:::{dropdown} Show full solution
```python
x_start = np.array([-1.0, -1.0])
design_screen = pyDOE3.ff2n(2) * 0.3 + x_start
df_screen = pd.DataFrame(design_screen, columns=['x1', 'x2'])
df_screen['y'] = [photocatalysis(r.x1, r.x2, noise_std=1.0) for r in df_screen.itertuples()]

fo = smf.ols('y ~ x1 + x2', data=df_screen).fit()
direction = np.array([fo.params['x1'], fo.params['x2']])
direction /= np.linalg.norm(direction)
endpoint = x_start + direction * 0.3 * 3   # walk 3 steps of length 0.3

def neg_photo_ep(x):
    return -photocatalysis(x[0], x[1], noise_std=1.0)

res_seq = minimize(neg_photo_ep, x0=endpoint, method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 0.01, 'fatol': 0.05})
print('Ascent endpoint:', endpoint.round(3))
print('NM from endpoint: nfev =', res_seq.nfev, ' value =', round(-res_seq.fun, 2))
print('(NM from raw corner, Section 21.3: nfev = 519, value = 60.39)')
```
Output: the 4-run screening factorial's direction walks the starting point
to **(−0.343, −0.385)**; Nelder-Mead started from there converges to
**79.58%** — far closer to the true 82.6% maximum than starting cold from
the corner (60.39%) — but it needed **762 evaluations**, actually *more*
than the 519 used starting from the corner directly (766 including the 4
screening runs). The trade-off here is real, not a clean win either way: a
short screening step before the simplex gets you a noticeably better
*answer*, at the cost of a comparable or slightly higher total evaluation
budget — a fair price when the answer quality matters more than the run
count, and exactly the kind of practical trade-off a real experimenter
weighs before choosing where to start a simplex search.
:::

## Notebook 22 (Live Tutorial 1): Two-Factor DoE, Start to Finish

**Exercise 1 — Extend to a third factor**

> **Hint:** Add `x3` (fibre volume fraction) to `pyDOE3.ff2n(3)`, replicate
> the 8 corners twice plus 3 centre points exactly as Section 22.1 does for
> two factors, give `x3` a real main effect but *no* interaction with the
> others, and rerun the same `smf.ols('ILSS_MPa ~ x1*x2*x3', ...)` and
> curvature-test code unchanged.

:::{dropdown} Show full solution
```python
corners3 = pyDOE3.ff2n(3)
design3 = pd.DataFrame(np.vstack([corners3, corners3]), columns=['x1','x2','x3'])
centre3 = pd.DataFrame({'x1':[0,0,0],'x2':[0,0,0],'x3':[0,0,0]})
design3 = pd.concat([design3, centre3], ignore_index=True)

xa, xb, xc = design3['x1'], design3['x2'], design3['x3']
true_ILSS3 = 45 + 6*xa + 3*xb + 2*xa*xb + 2.5*xc     # x3 = fibre volume fraction, main effect only
design3['ILSS_MPa'] = (true_ILSS3 + rng.normal(0, 1.2, len(design3))).round(2)

model_full3 = smf.ols('ILSS_MPa ~ x1*x2*x3', data=design3).fit()
print(model_full3.params.round(3))
print(model_full3.pvalues.round(4))
```
Output: `x3`'s coefficient comes out at **2.56** (true 2.5), significant
at p<0.0001 — the new factor's main effect is correctly identified,
exactly as confidently as `x1` and `x2`. Every interaction *involving*
`x3` (`x1:x3`, `x2:x3`, `x1:x2:x3`) correctly fails to reach significance
(p=0.89, 0.89, 0.17), matching the built-in truth that fibre volume
fraction was given no real interaction with anything else. Running
Section 22.4's curvature test on this 3-factor centre point also correctly
finds no significant curvature (F=0.49, p=0.556) — a straight-line-plus-
interaction model remains adequate, exactly as it should when no quadratic
term was ever built into the simulation. This is the exact groundwork
Notebook 24 (Live Tutorial 3) picks up with a full 5-factor screening
study.
:::

**Exercise 2 — Curvature, deliberately added**

> **Hint:** Add `- 3 * x1**2` to `true_ILSS` and rerun Section 22.4's
> F-test unchanged — but check the p-value carefully rather than assuming
> it clears 0.05, since this design's pure-error estimate comes from only
> 3 centre points (2 degrees of freedom), a genuinely weak test.

:::{dropdown} Show full solution
```python
x1, x2 = design_full['x1'], design_full['x2']
true_ILSS_curved = 45 + 6*x1 + 3*x2 + 2*x1*x2 - 3*x1**2
design_full['ILSS_MPa'] = (true_ILSS_curved + rng.normal(0, 1.2, len(design_full))).round(2)

model_full = smf.ols('ILSS_MPa ~ x1*x2', data=design_full).fit()
center_runs = design_full[(design_full[['x1','x2']]==0).all(axis=1)]
corner_runs_all = design_full[(design_full[['x1','x2']]!=0).all(axis=1)]
y_center_mean, y_corner_mean = center_runs['ILSS_MPa'].mean(), corner_runs_all['ILSS_MPa'].mean()
predicted_at_center = model_full.params['Intercept']

ss_curv = (len(corner_runs_all)*len(center_runs)/(len(corner_runs_all)+len(center_runs))) * \
          (y_corner_mean - y_center_mean)**2
ss_pe = ((center_runs['ILSS_MPa'] - y_center_mean)**2).sum()
df_pe = len(center_runs) - 1
F = ss_curv / (ss_pe/df_pe)
p = 1 - stats.f.cdf(F, 1, df_pe)
print(f'centre mean={y_center_mean:.2f}, model intercept={predicted_at_center:.2f}, F={F:.2f}, p={p:.4f}')
```
Output: with the true `-3*x1**2` curvature term added, **F=16.80,
p=0.055** — the centre-point mean (45.21) now noticeably exceeds the
model's linear prediction at (0,0) (43.02), but the test lands just
*barely* on the non-significant side of the conventional α=0.05 line. This
is not a failure of the test — it is an honest demonstration of its
limited power with only 2 pure-error degrees of freedom: doubling the true
curvature coefficient to −5 gives F=46.5, p=0.021, clearly significant,
confirming the test *does* work correctly, it just needs a large enough
signal (or more centre-point replicates) to reliably detect a modest one.
The one-sentence answer for an engineer seeing p=0.055 in practice:
treat it as a genuine warning sign worth a follow-up RSM study (Notebook
19) rather than either dismissing it outright or over-reacting — a
p-value this close to the threshold, with only 2 df backing it, is exactly
the situation that calls for more data, not a hard yes/no verdict.
:::

**Exercise 3 — Confirmation run outside the interval**

> **Hint:** Just add a constant to the simulated confirmation array
> (`confirmation + 4`) and recheck it against the *same* prediction
> interval computed in Section 22.5 — the interval itself doesn't change,
> only whether the (now-drifted) confirmation mean still falls inside it.

:::{dropdown} Show full solution
```python
confirmation_drift = confirmation + 4   # simulate an unmodelled +4 MPa process drift
print('Drifted confirmation runs:', confirmation_drift.round(2))
print('Drifted mean:', round(confirmation_drift.mean(), 2))
print(f'95% PI: [{lo:.2f}, {hi:.2f}]')
print('Inside interval?', lo <= confirmation_drift.mean() <= hi)
```
Output: the drifted mean (**59.59 MPa**) still lands **just inside** the
original 95% prediction interval of **[54.09, 60.07]** — 0.48 MPa under
the upper bound. This is a genuinely useful edge case: a +4 MPa
unmodelled drift is *not* automatically caught by a 95% interval whose
own width happens to be almost 6 MPa, and a slightly larger drift (or a
narrower interval from a less noisy process) would have flagged it. As
the engineer, seeing a confirmation run land near the very edge of an
interval like this should prompt the same caution as one that falls
fully outside it — investigate whether something in the process has
shifted (a new batch of resin, a recalibrated furnace) before trusting the
model's recommendation for production, rather than treating "technically
inside the interval" as an automatic pass.
:::

**Exercise 4 — Design your own 2² study**

> **Hint:** Pick any process from earlier in the course (e.g. ZnO
> synthesis from Notebook 17) and follow Sections 22.1–22.5 verbatim,
> swapping in your own two factors, ranges, and a `true_response()`
> function with a genuine interaction term and *no* quadratic term (so the
> curvature test should correctly come back non-significant).

:::{dropdown} Show full solution
```python
# ZnO nanoparticle synthesis: x1 = sintering temperature, x2 = precursor ratio
corners = pyDOE3.ff2n(2)
design_zno = pd.DataFrame(np.vstack([corners, corners]), columns=['x1', 'x2'])
centre_zno = pd.DataFrame({'x1': [0, 0, 0], 'x2': [0, 0, 0]})
design_zno = pd.concat([design_zno, centre_zno], ignore_index=True)

x1, x2 = design_zno['x1'], design_zno['x2']
true_grain = 20 + 8*x1 - 3*x2 + 2*x1*x2      # genuine interaction, no quadratic term
design_zno['grain_nm'] = (true_grain + rng.normal(0, 1.2, len(design_zno))).round(2)
design_zno = design_zno.sample(frac=1, random_state=1).reset_index(drop=True)

model_zno = smf.ols('grain_nm ~ x1*x2', data=design_zno).fit()
print(model_zno.params.round(3))

effects = 2 * model_zno.params.drop('Intercept')
sig = model_zno.pvalues.drop('Intercept') < 0.05
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(effects.abs().sort_values().index, effects.abs().sort_values().values,
        color=['crimson' if sig[t] else 'steelblue' for t in effects.abs().sort_values().index])
ax.set_xlabel('|Effect| (nm)'); ax.set_title('Pareto chart — ZnO grain size (own study)')
plt.tight_layout()
plt.show()
```
This follows exactly the same five-step recipe as Sections 22.1–22.5 —
build the design, fit the full model, rank effects with a Pareto chart,
check curvature at the centre point, and finish with a confirmation run at
the predicted optimum — applied to a process from earlier in the course
instead of the composite panel. There is no single "correct" numeric
answer for a self-chosen system; the exercise is complete once the
end-to-end workflow runs and the fitted coefficients are close to the
values you deliberately built into `true_grain`.
:::

## Notebook 23 (Live Tutorial 2): From Minitab to Six Factors

**Exercise 1 — A different dominant factor**

> **Hint:** Only `true_density`'s coefficients need to change — swap A and
> B's sizes (make B's coefficient the largest) and zero out `A:C` — the
> rest of Sections 23.1–23.3's code (design generation, Pareto chart,
> 80%-cumulative reduction) runs completely unchanged.

:::{dropdown} Show full solution
```python
A, B, C = results1['A'], results1['B'], results1['C']    # results1 built the same way as results in 23.1
true_density1 = 92.0 + 1.0*A + 4.5*B + 2.0*C + 0*A*C      # B now dominant, A:C removed
Y1 = true_density1.values[:, None] + rng.normal(0, 0.3, size=(len(results1), 5))
for i in range(5):
    results1[f'Y{i+1}'] = Y1[:, i].round(2)
results1['y'] = results1[[f'Y{i+1}' for i in range(5)]].mean(axis=1)

mod_full1 = smf.ols('y ~ A*B*C', data=results1).fit()
effects1 = (2*mod_full1.params.drop('Intercept')).abs().sort_values(ascending=False)
cum_pct1 = (effects1.cumsum()/effects1.sum()*100).round(1)
print(pd.DataFrame({'abs_effect': effects1.round(3), 'cum_pct': cum_pct1}))

keep_terms1 = cum_pct1.index[cum_pct1.shift(fill_value=0) < 80].tolist()   # smallest set reaching 80 %
print('Terms kept (smallest set reaching 80%):', keep_terms1)
```
Output:
```
   abs_effect  cum_pct
B       8.987     58.1
C       3.919     83.5
A       2.034     96.6
...
```
`B` correctly comes out as the largest bar (8.987, matching its built-in
true effect of 4.5×2=9.0), confirming the Pareto chart correctly tracks
whichever coefficient is dominant, not literally "A" by position. The
smallest set reaching 80% is **`B` and `C`** — `C` takes the running
total past the line (83.5%) — but it drops **`A`** (2.034, true effect
2.0), a genuine factor. As in Section 23.3, the 80% line is a first look,
not a decision rule: where the running total happens to cross 80% decides
what is kept, not the noise. With the 40 individual pellets (Section
23.2), `A`'s true coefficient of 1.0 against a pellet scatter of σ = 0.3
would be flagged easily.
:::

**Exercise 2 — Repeat for hue (y2) or brightness (y3)**

> **Hint:** The contrast loop in Section 23.5 and the auto-generated
> formula in Section 23.6 both work on any column name — just swap `'y1'`
> for `'y3'` (or `'y2'`) throughout, then look at the top few `|effect|`
> values to decide which terms belong in your own reduced model.

:::{dropdown} Show full solution
```python
effects_y3 = {}
for k in [1, 2, 3, 4, 5, 6]:
    for combo in labels[k]:
        factors = (combo,) if k == 1 else combo
        sign = np.prod([doe[f] for f in factors], axis=0)
        effects_y3[combo] = (sign * doe['y3']).sum() / 32

effects_series3 = pd.Series(effects_y3)
top10 = effects_series3.abs().sort_values(ascending=False).head(10)
print(top10)

top3_terms = [':'.join(t) if isinstance(t, tuple) else t for t in top10.index[:3]]
mod_y3 = smf.ols('y3 ~ ' + ' + '.join(top3_terms), data=doe).fit()
print(mod_y3.summary().tables[1])
print('R2 =', round(mod_y3.rsquared, 3), ' R2_adj =', round(mod_y3.rsquared_adj, 3))
```
Output: `y3`'s largest effects are **`x6` (8.875)**, **`x4` (3.000)**, and
a **4-way interaction `x2:x3:x4:x6` (2.875)** — noticeably messier than
`y1`'s clean 3-main-effects story, since a real interaction beats several
simpler two-factor terms here. Fitting `y3 ~ x6 + x4 + x2:x3:x4:x6` gives
R²=0.488 (R²_adj=0.462), all three terms significant (p≤0.030) but a
markedly weaker fit than `y1`'s reduced model (R²_adj=0.565) — this
response is simply harder to explain with a small term set. Two of `y1`'s
three dominant factors (`x4`, `x6`) matter for `y3` too; `x1` (important
for `y1`) drops out of `y3`'s picture entirely, replaced by a genuine
higher-order interaction — a reminder that "which factors matter" is a
question that has to be asked separately for every response you measure,
not assumed to transfer from one to another.
:::

**Exercise 3 — When does the formula-generation trick stop being optional?**

> **Hint:** No simulation needed here — just evaluate $2^k-1$ for
> $k=6,8,10$ and compare to the 63-term formula Section 23.6 already
> generated for 6 factors.

:::{dropdown} Show full solution
```python
for k in [6, 8, 10]:
    print(f'k={k}: 2^k - 1 = {2**k - 1} terms')
```
Output: **63** terms for 6 factors (already unwieldy by hand, which is
exactly why Section 23.6 writes code to generate the formula string), but
**255** for 8 factors and **1023** for 10. Somewhere well before 255 terms
— arguably already at 63 — writing the formula by hand stops being merely
"tedious" and starts being "not realistically possible without a
transcription bug": at 255+ terms even careful manual entry is all but
guaranteed to drop or duplicate a term, whereas the generator code in
Section 23.6 scales to any `k` with zero additional risk.
:::

**Exercise 4 — A fractional alternative**

> **Hint:** A genuine half-fraction of 6 factors (`I=ABCDEF`, a single
> 6-letter generator) is actually **Resolution VI**, not IV — to build a
> true Resolution IV design for 6 factors you need a *quarter* fraction
> (two generators, 16 runs, e.g. `E=ABC, F=BCD`). Try both, merging each
> fraction's rows back onto the full 64-row `doe` table (`pd.merge` on the
> `x1..x6` columns) to reuse the real Box & Draper response values.

:::{dropdown} Show full solution
```python
# Half-fraction, I=ABCDEF (actually Resolution VI, better than requested)
design_half = pyDOE3.fracfact('a b c d e abcde')
dfh = pd.DataFrame(design_half, columns=[f'x{i+1}' for i in range(6)])
merged_half = dfh.merge(doe, on=[f'x{i+1}' for i in range(6)], how='left')
mod_half = smf.ols('y1 ~ x1+x4+x6', data=merged_half).fit()
print('Half-fraction (32 runs):'); print(mod_half.params.round(3)); print(mod_half.pvalues.round(4))

# True Resolution IV quarter-fraction, generators E=ABC, F=BCD
design_quarter = pyDOE3.fracfact('a b c d abc bcd')
dfq = pd.DataFrame(design_quarter, columns=[f'x{i+1}' for i in range(6)])
merged_q = dfq.merge(doe, on=[f'x{i+1}' for i in range(6)], how='left')
mod_q = smf.ols('y1 ~ x1+x4+x6', data=merged_q).fit()
print('\nQuarter-fraction (16 runs):'); print(mod_q.params.round(3)); print(mod_q.pvalues.round(4))
```
Output: the full-data reduced model (all 64 runs) gave `x1`=0.873,
`x4`=1.492, `x6`=1.345, all p≤0.001. The **32-run half-fraction**
recovers `x1`=0.944, `x4`=1.744, `x6`=1.694, all still significant
(p≤0.010) — but note this "half-fraction" turns out to be **Resolution
VI** ($I=ABCDEF$, a single 6-letter defining word), not Resolution IV as
the exercise names it; a genuine half-fraction of 6 factors is always the
*maximum* achievable resolution, one better than IV. The **16-run quarter
fraction** (a true Resolution IV design, generators $E=ABC$, $F=BCD$)
still recovers all three terms as significant too (`x1`=1.619, p=0.014;
`x4`=2.069, p=0.003; `x6`=1.231, p=0.049) — `x6` only barely clears 0.05
at this smaller sample size, a direct consequence of fewer runs leaving
less power, exactly the trade-off Notebook 18 predicts. Neither fraction
here happens to alias `x1`, `x4`, or `x6` with each other in a way that
biases their estimates, since the Box & Draper system's dominant terms
are all main effects rather than interactions with each other.
:::

## Notebook 24 (Live Tutorial 3): Model Reduction and Fractional Designs

**Exercise 1 — Design efficiency comparison**

> **Hint:** `model_res3` (already fit on `df_res3` in Section 24.3) already
> has everything you need — look at its full p-value table, not just the
> `D` coefficient the section highlights, and remember `D=AB` on this
> design when deciding where `A:B`'s effect "went."

:::{dropdown} Show full solution
```python
print(model_res3.summary().tables[1])
```
Output:
```
              coef  std err       t  P>|t|
Intercept  17.7037    0.722  24.537  0.002
A           4.1762    0.722   5.788  0.029
B           6.0237    0.722   8.349  0.014
C           3.7287    0.722   5.168  0.035
D          -0.7388    0.722  -1.024  0.414
E          -1.0338    0.722  -1.433  0.288
```
With only 8 runs (2 residual df), `A`, `B`, `C` still reach significance
(p≤0.035), but `D` and `E` do **not** (p=0.414, 0.288) — even though both
have real, nonzero true coefficients (−2.0 and −1.0). `D`'s failure has two
compounding causes: it is aliased with the genuinely real `A:B`
interaction (true +2.0), which pulls its estimate from −2.0 toward
$-2.0+2.0=0.0$ (fitted: −0.739, "hiding" the missing A:B signal inside
what looks like a weak D effect) — and with only 2 residual degrees of
freedom, standard errors are large enough that even `E`'s unbiased −1.0
estimate can't clear the significance bar either. `A:B`'s own effect,
specifically, is inseparably blended into the `D` estimate on this design
— there is no way to recover it as its own number without more runs (see
Exercise 3's fold-over).
:::

**Exercise 2 — A different aliasing pattern**

> **Hint:** `pyDOE3.fracfact('a b c bc ac')` gives `D=BC`, `E=AC` instead of
> `D=AB`, `E=AC` — check which of `B:C` and `A:C` are actually part of the
> true model (`true_strength` only ever uses `A:B` and `C:D` as real
> interactions), then predict whether `D` and `E` will come out biased or
> clean before running the fit.

:::{dropdown} Show full solution
```python
design_res3b = pyDOE3.fracfact('a b c bc ac')
df_res3b = pd.DataFrame(design_res3b, columns=['A','B','C','D','E'])
print('D == B*C:', np.allclose(df_res3b['D'], df_res3b['B']*df_res3b['C']))
print('E == A*C:', np.allclose(df_res3b['E'], df_res3b['A']*df_res3b['C']))

true_strength_b = (18 + 4*df_res3b.A + 6*df_res3b.B + 3*df_res3b.C - 2*df_res3b.D - 1*df_res3b.E
                    + 2*df_res3b.A*df_res3b.B - 1.5*df_res3b.C*df_res3b.D)
df_res3b['strength_MPa'] = (true_strength_b + rng.normal(0, 1.0, len(df_res3b))).round(2)

model_res3b = smf.ols('strength_MPa ~ A+B+C+D+E', data=df_res3b).fit()
print(model_res3b.params.round(3))
```
Output: this generator gives `D=BC` and `E=AC` instead of Section 24.3's
`D=AB`, `E=AC`. Since `true_strength` never includes a real `B:C` term
(it's zero), `D` comes out essentially **unbiased**: fitted **−2.184**
against true **−2.0**. Likewise `A:C` is also zero in the true model, so
`E`'s fitted **−0.886** stays close to its true **−1.0** too. The contrast
with Section 24.3's design is the whole point: **the same Resolution III
penalty (main effects aliased with 2FIs) does not always cause visible
bias** — it depends entirely on whether the *specific* 2FI a main effect
happens to be aliased with is one of the real, nonzero interactions in the
true (unknown, in a real experiment) system. A design's resolution tells
you the *risk* is present; it cannot tell you in advance whether that risk
will actually bite for any particular effect.
:::

**Exercise 3 — Fold-over**

> **Hint:** `-1 * design_res3` gives the fold-over runs; simulate a fresh
> response on them using the *same* true model, `pd.concat` both halves
> into one 16-run table, then fit a model that includes both `D` and
> `A:B` together — on the combined design they are no longer the same
> column, so both should recover cleanly.

:::{dropdown} Show full solution
```python
design_res3_fold = -1 * design_res3
df_res3_fold = pd.DataFrame(design_res3_fold, columns=['A','B','C','D','E'])
true_strength_fold = (18 + 4*df_res3_fold.A + 6*df_res3_fold.B + 3*df_res3_fold.C
                       - 2*df_res3_fold.D - 1*df_res3_fold.E
                       + 2*df_res3_fold.A*df_res3_fold.B - 1.5*df_res3_fold.C*df_res3_fold.D)
df_res3_fold['strength_MPa'] = (true_strength_fold + rng.normal(0, 1.0, len(df_res3_fold))).round(2)

combined = pd.concat([df_res3, df_res3_fold], ignore_index=True)
model_combined = smf.ols('strength_MPa ~ A+B+C+D+E+A:B+C:D', data=combined).fit()
print(model_combined.params.round(3))
print(model_combined.pvalues.round(4))
```
Output: on the combined 16-run design, **every** term is now separately
estimable and significant (all p≤0.009): `D=-2.145` (true −2.0),
`A:B=1.406` (true 2.0), `C:D=-1.385` (true −1.5). Folding over and
combining doubles the run count (8→16) but buys back exactly what
Resolution III cost: `D` and `A:B` are no longer confounded with each
other, so both can be estimated as their own separate numbers — a direct,
practical demonstration of the fold-over trick Notebook 18's Exercise 1
introduces in the abstract.
:::

## Notebook 25 (Live Tutorial 4): Optimization and Mixture Designs

**Exercise 1 — A 4-component mixture**

> **Hint:** `simplex_lattice(q=4, m=2)` needs no new code — just call it
> with `q=4` and count the rows; compare that count directly to `2**4`
> for the standard-factorial alternative the introduction contrasts it
> with.

:::{dropdown} Show full solution
```python
lattice_42 = simplex_lattice(q=4, m=2)
print(f'{{4,2}} simplex-lattice points: {len(lattice_42)}  (vs 2^4 factorial: 16 runs)')
print(pd.DataFrame(lattice_42, columns=['EC', 'EMC', 'DMC', 'FEC']))
```
Output: the **{4,2}** simplex-lattice design has **10 points** — the 4
pure-component vertices plus the $\binom{4}{2}=6$ binary-blend edge
midpoints — versus 16 runs for a $2^4$ factorial that would mostly
describe physically impossible blends anyway. Adding a 4th solvent barely
grows the mixture design (6→10 points) even though a factorial on the
same 4 "factors" would double in size, which is the practical payoff of
working directly in the constrained simplex rather than trying to force a
cube onto a proportion constraint.
:::

**Exercise 2 — Constrained mixture**

> **Hint:** `simplex_lattice(q=3, m=10)` gives a fine 66-point grid over
> the whole triangle — boolean-mask it with
> `(grid['EC'] >= 0.2) & (grid['EC'] <= 0.4)` and just count what survives.

:::{dropdown} Show full solution
```python
grid_full = pd.DataFrame(simplex_lattice(q=3, m=10), columns=['EC','EMC','DMC'])
print('Unconstrained {3,10} lattice points:', len(grid_full))

constrained = grid_full[(grid_full['EC'] >= 0.2) & (grid_full['EC'] <= 0.4)]
print('Points with 0.2 <= EC <= 0.4:', len(constrained))
print(constrained.head())
```
Output: **66** unconstrained lattice points shrink to **24** once the
0.2–0.4 EC band is enforced — a band running as a narrow horizontal strip
across the ternary triangle rather than the whole simplex. That reshaped,
band-limited feasible region is exactly what a real "constrained mixture
design" tool (D-optimal, beyond this course) is built to search
efficiently — filtering a fine regular grid, as done here, is a
reasonable low-tech approximation for a small number of components, but
it stops scaling once you have more components or a tighter combination
of several such constraints at once.
:::

**Exercise 3 — Special cubic model**

> **Hint:** Add `EC:EMC:DMC` to the model formula (`df_mix['EC_EMC_DMC'] =
> df_mix.EC * df_mix.EMC * df_mix.DMC`, then include it in the `smf.ols`
> formula) and look at the residual degrees of freedom before trusting any
> p-value — with only 9 runs and now 7 parameters, there isn't much slack
> left.

:::{dropdown} Show full solution
```python
df_mix['EC_EMC_DMC'] = df_mix.EC * df_mix.EMC * df_mix.DMC
model_special = smf.ols(
    'conductivity_mS_cm ~ EC+EMC+DMC+EC:EMC+EC:DMC+EMC:DMC+EC:EMC:DMC - 1',
    data=df_mix
).fit()
print('Residual df:', int(model_special.df_resid))
print(model_special.pvalues.round(4))

lattice_33 = simplex_lattice(q=3, m=3)
print(f'\n{{3,2}} points: {len(lattice_32)}, {{3,3}} points: {len(lattice_33)}')
interior = sum(1 for p in lattice_33 if np.all(p > 0) and np.all(p < 1))
print('Genuinely interior points in {3,3} (none in {3,2}):', interior)
```
Output: with **9 runs and 7 parameters**, only **2 residual degrees of
freedom** remain, and **none** of the interaction terms stay significant
— not even `EC:EMC` and `EC:DMC`, which *were* clearly significant in
Section 25.4's simpler 6-parameter model (p=0.113 and 0.109 here, both
well above 0.05; `EC:EMC:DMC` itself comes in at p=0.857). The design is
essentially saturated: adding one more parameter without adding more runs
leaves too little residual information to trust any single coefficient,
including ones that were solid before. The {3,2} lattice used throughout
this notebook has only 6 points (3 vertices + 3 edge midpoints) with zero
genuinely interior points; {3,3} adds 10 points total but only **1** of
them (the centroid, 1/3-1/3-1/3) is actually interior — a real
special-cubic study needs a design like {3,3} or larger specifically
*because* it needs at least one interior point (and ideally several,
replicated) to separate the special-cubic term from the ordinary
quadratic interactions at all.
:::

**Exercise 4 — Optimize under a lower bound**

> **Hint:** Reuse Section 25.5's `grid_df` (already has a `pred` column
> from `model_mix.predict`) and just mask it with `grid_df['EC'] >= 0.15`
> before taking `.idxmax()` — compare the masked and unmasked maxima
> directly.

:::{dropdown} Show full solution
```python
best_unconstrained = grid_df.loc[grid_df['pred'].idxmax()]
constrained_grid = grid_df[grid_df['EC'] >= 0.15]
best_constrained = constrained_grid.loc[constrained_grid['pred'].idxmax()]

print('Unconstrained optimum:', best_unconstrained.round(4).to_dict())
print('Constrained (EC>=0.15) optimum:', best_constrained.round(4).to_dict())
print('Conductivity given up:', round(best_unconstrained['pred'] - best_constrained['pred'], 4), 'mS/cm')
```
Output: the constrained and unconstrained optima are **identical** —
both land at **EC=0.15, EMC=0.85, DMC=0.00**, giving up **0.0 mS/cm**.
This isn't a bug in the filter: Section 25.5's own unconstrained search
already reported EC=0.15 as the best blend (the bright region hugs the
EC–EMC edge right down to the low end of EC, per that section's
take-home message), so the practical minimum-EC requirement here happens
not to bind at all — according to the *fitted* model, the constraint costs
nothing. But Section 25.5 shows that the fitted optimum is misleading: the
true best blend is pure DMC (6.0 mS/cm), which EC ≥ 0.15 excludes. With
the true coefficients, the best blend satisfying the constraint is EC 0.15
/ DMC 0.85 at 5.93 mS/cm — on a different edge from the fitted model's
answer — so the constraint really costs about 0.07 mS/cm. A constraint
that looks free under an uncertain model may not be; check the answer
against the model's uncertainty (Section 25.5) before relying on it.
:::

**Exercise 5 — Steepest ascent, followed through**

> **Hint:** Replace the raw coded coordinates in `true_adhesive_strength`
> with a soft-saturating transform (`1.5*np.tanh(x/1.5)` behaves linearly
> near 0 but flattens out for large |x|, standing in for "a real process
> limit") before evaluating it along the extended path — then look at how
> the *marginal* gain per step (`.diff()`) shrinks, not just the raw
> response values.

:::{dropdown} Show full solution
```python
def true_adhesive_extended(df):
    Ae, Be, Ce = 1.5*np.tanh(df.A/1.5), 1.5*np.tanh(df.B/1.5), 1.5*np.tanh(df.C/1.5)
    De, Ee     = 1.5*np.tanh(df.D/1.5), 1.5*np.tanh(df.E/1.5)
    return 18 + 4*Ae + 6*Be + 3*Ce - 2*De - 1*Ee + 2*Ae*Be - 1.5*Ce*De

n_steps = 20
path_ext = pd.DataFrame([start.values + direction_normalized.values * step_size * i
                         for i in range(n_steps)], columns=list('ABCDE'))   # Section 25.1's path, from the corner
path_ext['y_true'] = true_adhesive_extended(path_ext)
path_ext['delta'] = path_ext['y_true'].diff()
initial_gain = path_ext['delta'].iloc[1]
path_ext['pct_of_initial_gain'] = (path_ext['delta'] / initial_gain * 100).round(1)

first_below_half = path_ext[path_ext['pct_of_initial_gain'] < 50].index[0]
print('Initial marginal gain per step:', round(initial_gain, 3), 'MPa')
print('First step where gain < 50% of initial:', first_below_half)
print(path_ext[['y_true', 'delta', 'pct_of_initial_gain']].round(2))
```
Output: starting from Section 25.1's best corner (34.66 MPa on this
saturating version of the process), the per-step marginal gain starts at
**3.35 MPa/step** and falls below **50% of that initial rate at step 4**
(1.61 MPa/step, 48.1% of the initial gain) — well before the raw response
stops climbing altogether (it is still gaining, just more and more
slowly, at step 19). For comparison, the classic path from the centre
(4.21 MPa/step at first) reaches the same point only at step 7 (1.82
MPa/step, 43.3%), at a lower strength (40.7 vs. 44.5 MPa): the corner
start saves three experiments. Because a
`tanh` saturation never actually reverses (it approaches a limit, it
doesn't peak and fall), there is no single unambiguous "stop here" step
the way a true quadratic optimum would give you; the 50%-of-initial-gain
threshold is a reasonable, defensible proxy for "diminishing returns have
become impossible to ignore," and step 4 is where an engineer running this
sequence in real life would notice each additional experiment buying
markedly less than the first few did — precisely the moment Notebook 22
(Live Tutorial 1) §22.4's curvature test is designed to formalise, and
the moment it's time to stop walking a straight line and switch to a
proper quadratic RSM study (Notebook 19).
:::

**Exercise 6 — Your own mixture study**

> **Hint:** Pick a genuinely constrained materials system (a glass
> composition summing to 100 mol%, an alloy, a polymer blend) and follow
> Sections 25.3–25.5's own three function calls — `simplex_lattice`
> (or `simplex_centroid`), the `- 1` no-intercept Scheffé formula, and the
> `ternary_to_cartesian` plot — end to end with your own plausible
> pure-component values and at least one synergy or antagonism term.

:::{dropdown} Show full solution
```python
# Soda-lime glass: SiO2 / Na2O / CaO, response = softening point proxy (deg C)
lattice_glass = simplex_lattice(q=3, m=2)
df_glass = pd.DataFrame(lattice_glass, columns=['SiO2', 'Na2O', 'CaO'])

b = {'SiO2': 1650, 'Na2O': 850, 'CaO': 1200,        # pure-component values
     'SiO2:Na2O': -350, 'SiO2:CaO': -150, 'Na2O:CaO': 50}   # blending terms
true_T = (b['SiO2']*df_glass.SiO2 + b['Na2O']*df_glass.Na2O + b['CaO']*df_glass.CaO
          + b['SiO2:Na2O']*df_glass.SiO2*df_glass.Na2O
          + b['SiO2:CaO']*df_glass.SiO2*df_glass.CaO
          + b['Na2O:CaO']*df_glass.Na2O*df_glass.CaO)
df_glass['T_soft_C'] = (true_T + rng.normal(0, 10, len(df_glass))).round(1)

model_glass = smf.ols(
    'T_soft_C ~ SiO2+Na2O+CaO+SiO2:Na2O+SiO2:CaO+Na2O:CaO - 1', data=df_glass
).fit()
print(model_glass.params.round(1))
```
Output (one valid run):
```
SiO2         1644.7
Na2O          862.2
CaO          1200.3
SiO2:Na2O    -375.8
SiO2:CaO     -160.4
Na2O:CaO       79.4
```
The fitted pure-component coefficients land close to the `b` dictionary's
built-in values (SiO₂≈1650, Na₂O≈850, CaO≈1200), exactly as Section 25.4's
checkpoint predicts: each Scheffé linear coefficient *is* the model's
predicted response for that pure component. There is no single "right"
system or "right" numbers for this exercise — the exercise is complete
once your own chosen materials system, `simplex_lattice`/`simplex_centroid`
design, no-intercept Scheffé formula, and (optionally) a ternary contour
plot (Section 25.5's `ternary_to_cartesian` two-liner) all run end to end
and the fitted coefficients recover whatever synergy or antagonism you
built into your own true model.
:::
