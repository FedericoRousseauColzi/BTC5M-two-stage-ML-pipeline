# BTC5M-two-stage-ML-pipeline

A short-horizon direction model for BTCUSDT on 5-minute bars, built in two stages: one network decides whether the current bar is a turning point worth trading, a second decides which way it resolves.

## The idea

Most ML on price tries to call direction on every bar, which is mostly noise. I wanted to split the problem: first find the small fraction of bars that look like local turning points, and only there ask "up or down?". Framing it as two easier questions is the whole point. The first model buys selectivity, the second gets to specialise. The first stage is a Conv1D+LSTM gate; the second is a GRU that also sees the gate's own confidence as an input, which is the meta-labelling idea from López de Prado's Advances in Financial Machine Learning.

It runs on about 2.5 years of 5-minute BTC data (Feb 2024 to Jul 2026). Out-of-sample it shows a small edge, I'll explain below why I'm not making a big deal of it.


## How it actually works

Before any features, the raw candles pass through an adaptive Kalman filter that denoises the price and produces a few extra signals (a smoothed velocity, a volatility estimate, and the filter's own "surprise", i.e. its innovations). That filter is the one piece I've kept private, so it ships as precomputed features rather than source. Everything downstream runs on its outputs, so the notebooks still reproduce end to end.
From there:

- **Stage 1, the gate**. I label local highs and lows as pivots (~1.5% of bars) and train a Conv1D×3 (dilated, causal) + LSTM(32) on 28 causal features to spot them. At a 0.86 probability cutoff, plus a small rule so it doesn't fire five bars in a row, it flags roughly 4% of bars.

- **Labels**. Each gated signal gets a triple-barrier label (±3.5×ATR, 120-bar cap). I compute it against both the smoothed and the raw price and keep the disagreement rate (~16%) as an honest measure of how much the smoothing is flattering the labels.

- **Stage 2, direction**. A GRU(203) + Dense(89) takes 30 features and calls long or short.

- **Calibration**. The raw GRU probabilities aren't well calibrated, so I fit an isotonic regression on the validation set and threshold the calibrated numbers (long ≥ 0.67, short ≤ 0.29). Isotonic only rescales the probabilities, it doesn't change the ranking.

- **Backtest**. Plain event-driven ATR take-profit / stop-loss, enter at the next bar's open, ties go to the stop, fees included.

## Results

Everything here is on a 90-day holdout (9 Apr – 7 Jul 2026) that the models never saw in training or validation.

Directional model:


| | |
|---|---|
| Trades taken (long ≥ 0.67 / short ≤ 0.29) | 54 (40L / 14S), ~5.5% of gated signals |
| Directional accuracy | 74.1% |
| Precision, long / short | 0.78 / 0.64 |
| Calibration | isotonic (Brier 0.245 → 0.242 on val) |

Event-driven backtest (net of round-trip fees):

| | |
|---|---|
| Trades | 61 (46L / 15S) |
| Win rate | 59.0% |
| Profit factor | 1.48 |
| Max drawdown | −3.29% |
| Sharpe (per-trade, annualised) | 2.49 |

<img width="452" height="362" alt="image" src="https://github.com/user-attachments/assets/5889791b-6066-402d-a94c-b5a87af8e47d" />


I'd rather be straight about these than dress them up:

- 61 trades isn't a lot. The t-stat on the average return is about 1.4 (so, not significant), and the 95% band on that annualised Sharpe is roughly [0, 6]. The point estimate is encouraging; it isn't proof. The Sharpe is computed on the per-trade returns, not a resampled equity curve, so it isn't inflated by flat bars.
- The gap between 74% label accuracy and a 59%-win rate is given by adjusting the accuracy for the raw/smoothed label disagreement where you'd expect a tradable precision around 60–65%, which is about what the backtest gives.
- The average edge is roughly 9 bps a trade, which is about what a round-trip costs. So the net edge is thin and cost sensitive. This is a research pipeline, not a money printer.

## Why there might be an edge at all

I don't think this is magic, and I'd be suspicious of anyone who did. Two stories I find plausible. Around local extremes a lot of the flow is liquidations and stop runs that partly reverse over the next hour or two, so catching the turn captures some of that snap-back. And the filter's innovations are basically a measure of when the market is surprising a simple model, which tends to coincide with regime shifts. Requiring both stages to agree means I only trade about 5% of candidate bars. My honest expectation is a small, cost-sensitive edge. Whether it survives on other assets, other regimes, and realistic fills is the open question.

## The filter (the omitted part)

<img width="452" height="272" alt="image" src="https://github.com/user-attachments/assets/e5d42945-3e64-45b4-b13a-1af3cace13b1" />

The panel above is what the filter does: raw candles on the right, denoised on the left. Its whole job is to strip microstructure noise and hand the models a cleaner price plus those few derived signals.

## What this doesn't prove

One asset, one timeframe, one ~2.5-year stretch that's essentially a single regime, and one 90-day test with a few dozen trades. The backtest's TP/SL levels and thresholds were chosen by looking at history, so there's some in-sample optimism baked in. The things I'd trust more than the current numbers, and the natural next steps: a proper walk-forward, a deflated Sharpe that accounts for the parameter search, a real fee and slippage sweep, and seeing whether any of this holds up on something other than BTC.
