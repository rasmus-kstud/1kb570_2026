Mastering DoE with Python
==========================

Laboration 3: Optimising a Catalytic Reaction — 4-Hour Contest
-----------------------------------------------------------------

## Learning goals

By the end of this session you will be able to:

- Design and analyse a multi-factor screening experiment, including a curvature check via centre points, and correctly drop a factor that turns out not to matter
- Use the factorial model to plan and execute a steepest-ascent search
- Design and fit a response-surface (RSM) model, and read a contour plot to locate an optimum
- Make deliberate trade-offs between the number of experiments you run and your confidence in the result
- Turn a point estimate into a formal, statistically defensible claim using a confidence interval, and reason about what more replicates would (and wouldn't) buy you

## Background

Earlier multivariate analysis identified reactant concentration **c** (mM) and temperature **T** (°C) as the two variables *most likely* to drive this catalytic reaction's rate — but it couldn't rule out a third suspect: **stirring rate** (rpm). The plant engineers have gone back and forth on whether agitation matters here; if the reaction is limited by how fast reactant reaches the catalyst surface (mass-transfer control), stirring harder should speed it up, but if the reaction is limited by the intrinsic chemistry at the catalyst (kinetic control), stirring speed above some minimum shouldn't matter at all. Nobody has settled this with real data. A previous "one variable at a time" (OVAT) optimisation effort landed the process at **c = 2.0 mM, T = 25.0 °C**, giving a reaction rate of **5.77 mol/s**. Management suspects this isn't the true optimum and wants it found — but running an experiment means shutting down the whole plant, so every run has a real cost.

## The contest

This is a **contest**, not just an exercise: the goal is to find the highest reaction rate you can, **using as few experiments as possible**. There is no fixed experiment budget — you can run as many as you want — but every single one is logged automatically, and a live leaderboard tracks each team's best rate found *and* how many experiments it took to get there. Efficiency counts as much as the result.

## The 4-hour workflow

| Time | Step | What you produce |
|---|---|---|
| 0:00–0:45 | **1. Factorial screening** | A 2³ factorial design (with centre points) around the OVAT point, analysed for significant effects and curvature — including whether stirring rate matters at all |
| 0:45–1:30 | **2. Steepest ascent** | A sequence of experiments moving in the direction of increasing rate, in whichever factors survived Step 1 |
| 1:30–3:00 | **3. Response surface (RSM)** | A Latin-hypercube design and fitted quadratic model, with a contour plot showing your optimum |
| 3:00–3:45 | **4. How sure are you?** | A replicated confirmation with a formal confidence interval, not just a single check |
| 3:45–4:00 | **Wrap-up** | Your answer sheet, finalised |

Stick to the timings loosely — the point is a complete, efficient search, not a perfect one.

## How to run experiments

Experiments run on a live "plant" server. Your lab instructor will give you a **server URL**, your **team name**, and your **team's token** at the start of the session. Open `plant_client.py`, fill in `SERVER_URL`, `TEAM`, and `TOKEN` at the top, then in your notebook:

```python
from plant_client import run_experiment, leaderboard

result = run_experiment(T=[25.0, 30.0], c=[2.0, 2.5], rpm=[500.0, 500.0], name='factorial')
result   # a DataFrame with columns T, c, rpm, rate
```
> NOTE: It is important that you **name your experiment**. This will store the data in a file so that when you re-run the cell, the function does not request more data. Every time you need more data, remember to give the experiment a new *unique* name.

Every call prints how many experiments your team has used **in total so far** — that number comes from the server, not from you, so it's the number that goes on your answer sheet (no honour-system counting, and no need to keep your own tally).

A few practical constraints, all enforced by the server:

- Up to **25 (T, c, rpm) triples per call** — the plant can only run one shutdown window's worth of experiments at a time, so plan each batch deliberately rather than submitting one point at a time.
- Valid ranges: **T in [0, 70] °C**, **c in [0.1, 6.0] mM**, **rpm in [100, 1000]** — outside this the reactor is considered unsafe and the server will reject the call.
- Every reading includes real replicate noise — repeat runs at the same conditions will not give identical rates, which is exactly why centre-point replicates in Step 1 (and the confidence interval in Step 4) matter.

Call `leaderboard()` any time to see every team's experiment count and best rate found so far (not their actual settings — find your own optimum).

## Dataset

There is no dataset to load this time — every data point you use comes from an experiment you actually ran. Keep a running table (a DataFrame you keep appending to) of every (T, c, rpm, rate) you've collected; you'll need the full history for Step 3, not just your most recent batch.

---

## Step 1 — Factorial Screening (0:00–0:45)

Three candidate factors go into this screen: **c**, **T**, and **stirring rate (rpm)** — the third is there because nobody has actually tested whether it matters, not because you're told the answer in advance.

1. Design a 2³ factorial (eight corner points) around the OVAT point (c = 2.0, T = 25.0) and a stirring rate you consider a sensible centre value (e.g. 500 rpm), plus a few centre-point replicates at that same centre. Choose your factor ranges deliberately — too small a step and effects may be lost in the noise; too large and you risk leaving the safe operating range or the local (linear) approximation breaking down.
2. Run the design via `run_experiment`.
3. **Code each factor to ±1** before fitting (e.g. `c_coded = (c - 2.0) / 0.5`) — with three factors and their interactions, fitting on raw units (mM, °C, rpm) gives a badly ill-conditioned model where every effect looks non-significant just from numerical collinearity, not because the effects aren't there. Coding fixes this and is standard factorial-design practice regardless.
4. Fit a linear model with interactions (`rate ~ c_coded * T_coded * rpm_coded`, `statsmodels.formula.api.ols`) to the corner points. Which effects are statistically significant?
5. If rpm (and any interaction involving it) is not significant, **drop it** — fix it at a convenient value for the rest of the lab and reason about *why* that makes physical sense for a catalytic reaction, rather than just reporting a p-value.
6. Compare the centre-point average to the corner-point average — a real difference indicates curvature (a hint that you're already close to the optimum, or that the local-linear approximation is starting to break down).

> **Checkpoint 1a**: Describe your factorial design (factor ranges for all three factors, number of corner points, number of centre-point replicates) and report which effects came out statistically significant, with their signs. If you dropped a factor, say which one, and give a physical reason a catalytic reaction could plausibly behave that way (think about what stirring rate actually controls in a stirred reactor).

> **Checkpoint 1b**: Did you detect significant curvature at the OVAT point (in the factors you kept)? What does that tell you about where the OVAT point sits relative to the true optimum, and how does it shape your Step 2 plan?

---

## Step 2 — Steepest Ascent (0:45–1:30)

Using the significant main effects from Step 1 (c and T, assuming rpm was dropped), compute the direction of steepest ascent and run a short sequence of experiments stepping along it (a common approach: pick a convenient step size in one variable, scale the other variable's step by the ratio of the fitted effects, and keep moving until the rate stops improving). Keep rpm fixed at the value you settled on in Step 1 — every `run_experiment` call still needs an `rpm` argument, it just shouldn't be varying anymore.

> **Checkpoint 2a**: Describe your steepest-ascent path — the points you tried and the rates you found — and explain when and why you stopped.

> **Checkpoint 2b**: How many experiments have you used in total so far (screening + ascent)? Check your number against the server's count from your last `run_experiment` call.

---

## Step 3 — Response Surface Optimisation (1:30–3:00)

Around the best region found in Step 2, design a set of experiments suited to fitting a **quadratic** model (a Latin-hypercube design is suggested — `scipy.stats.qmc.LatinHypercube` — since it spreads points efficiently through the region without the run count of a full factorial-plus-axial design). Fit `rate ~ c + T + I(c**2) + I(T**2) + c:T` and use it to locate the optimum.

1. Design and run your points.
2. Fit the quadratic model.
3. Produce a **contour plot** of predicted rate over the (c, T) region you explored, marking your best experimental point and the model's predicted optimum.
4. Confirm the predicted optimum with one or two additional experiments at (or near) the predicted (c*, T*) — a first, informal sanity check. Step 4 turns this into a real statistical claim.

> **Checkpoint 3a**: Describe your RSM design (how many points, what design, over what region) and report the fitted quadratic model's key terms.

> **Checkpoint 3b**: Report your predicted optimum (T*, c*, predicted rate) alongside the contour plot.

---

## Step 4 — How Sure Are You? A Confidence Interval on the Optimum (3:00–3:45)

Step 3's one or two confirmation runs only tell you the prediction wasn't *wildly* wrong — a single extra data point can't tell you how confident to be. Turn it into a real statistical claim.

1. Run **8–10 replicate experiments** at your predicted optimum (T*, c*, fixed rpm) — same settings every time.
2. Compute the sample **mean** and **standard deviation** of the resulting rates.
3. Build a **95% confidence interval** on the true mean rate at that point (the same one-sample CI construction from Part III — `scipy.stats.t.ppf` for the critical value, since you're estimating the population standard deviation from a small sample).
4. Check two things against your CI: does the OVAT baseline (5.77 mol/s) fall inside or outside it, and does your RSM model's *predicted* rate from Checkpoint 3b fall inside or outside it?

> **Checkpoint 4a**: Report your CI (n replicates, mean, standard deviation, and the 95% interval itself). Does the model's predicted rate from Checkpoint 3b fall inside your CI? What does that tell you about how much you can trust the model's point prediction, versus just eyeballing Step 3's one-off confirmation run?

> **Checkpoint 4b**: A confidence interval's half-width shrinks roughly with $1/\sqrt{n}$. Using your own n and half-width, estimate roughly how many replicates you'd need to halve it. Given the contest rewards using *fewer* experiments, is that trade worth making here — would you actually spend those extra experiments, or is your current interval already tight enough to act on? Justify briefly either way.

> **Checkpoint 4c**: Report the **total number of experiments** your team used across all four steps (from the server, not a self-count), and the **percentage improvement** in rate over the OVAT baseline of 5.77 mol/s. Looking back across all four steps, which one gave you the most rate improvement per experiment spent?

---

## What to hand in

**No lab report.** Hand in two things:

1. **Your working notebook** (`.ipynb`) — the code for all four steps, in order, with light comments. You do not need prose explanations here — that's what the answer sheet is for.
2. **Your answer sheet** — direct, concise answers to Checkpoints 1a through 4c (9 questions). A few sentences per answer is normal; a paragraph is too long. Include the specific numbers and plots each question asks for.

Names of everyone who contributed go on the answer sheet.
