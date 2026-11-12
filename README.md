# Roblox Up-and-Coming top-10 forecast

Entry for make.fun competition `roblox-up-and-coming-forecast-01`
(Predict Roblox's Up-and-Coming top 10, $500, deadline 2026-11-13 06:00 UTC).

## Program

`main.py` (Python 3.13, standard library only). It reads
`$MAKEFUN_INPUT_DIR` (else `/input`): `history.ndjson` + `task.json`, and
writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else `/output/forecast.json`):
one probability per game per place (1–10) plus an ordered `top10`.

## Model

Monte-Carlo simulation of the chart's rank dynamics (800 runs x 72 hours,
fixed seed, fully deterministic):

- **State:** each game's latent rank, starting at its last observed rank.
- **Drift per hour:** `0.6 * momentum + 0.05 * (-standardised 24 h playing
  growth) + 0.02 * (natural rank - rank)`, plus `N(0, 0.8)` noise; momentum
  decays 5% per hour. Momentum is the recency-weighted rank velocity over
  the last 4 reads; the natural rank is the recency-weighted mean rank over
  the game's history.
- **Aging-out:** a game whose `details.created` is known leaves the chart
  28 days after publication (the chart only lists games published in the
  last 28 days).
- **Probabilities:** the simulated frequency of each (game, place) pair,
  rescaled so 0.25 of every place stays unnamed (for games that join the
  chart during the window), floored at 0.0001, at most 40 games per place.

## Why this beats the house persistence baseline

Measured on the practice window with a local scorer that replicates the
rules' log score exactly (validated: it reproduces the baseline's published
-43.512039):

| forecast | log score | typical p | top-10 hits | agreement |
|---|---|---|---|---|
| house baseline (persistence) | -43.512 | 0.013 | 6.4 | 53.5 |
| this entry | **-33.064** | 0.037 | 6.4 | 54.6 |

A backtest on an earlier split (train 24 reads -> next 9 reads) gives
-45.9 vs the baseline's -56.9, so the edge is not fitted to one window.
The history shows short-term playing growth is noise for rank changes
(corr -0.02, mostly time-of-day) while 24 h growth predicts them
(corr -0.27) and rank velocity persists (autocorr 0.20) — the model uses
exactly those signals, with weights tuned on the practice window
(time-based recency windows and larger growth weights tested worse).

## Local test

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# run twice and diff: outputs are byte-identical; ~7 s per run; ~11 KB output
```

## License

MIT (see LICENSE).
