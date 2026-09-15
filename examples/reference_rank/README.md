# Why reference choices change a rank score

A rank compares a value with reference values. In a forecast comparison, each reference record supplies a forecast, an observed outcome and a baseline value. Those three values can differ, so one record can move their ranks in different directions.

Here is a small exact example. A record is described by a number `U`, equally likely to be 0, 1 or 2. Its forecast and outcome both equal `U`; its baseline equals `2−U`. Fix the record being evaluated at `U=1`. Its forecast, outcome and baseline are therefore all 1.

Give a comparison rank 1 when the evaluated value exceeds its reference, 0 when it is lower, and 1/2 for a tie. Subtract the baseline rank from the forecast rank to obtain `A`, and from the outcome rank to obtain `B`.

| Random reference `U` | Its forecast / outcome / baseline | Forecast contrast `A` | Outcome contrast `B` |
|---|---|---:|---:|
| 0 | 0 / 0 / 2 | 1 | 1 |
| 1 | 1 / 1 / 1 | 0 | 0 |
| 2 | 2 / 2 / 0 | −1 | −1 |

Using the same reference makes the average product `A×B` equal `2/3`. Using two independent references makes it 0: each contrast has mean 0. The evaluated values stay fixed; the reference assignment changes the score's expectation.

This calculation conditions on the evaluated record being `U=1`. Averaging over all three possible evaluated records gives a population target of `8/27`, an expected shared score of `2/3`, and a reference interaction of `10/27`. The example illustrates comparison arithmetic; it makes no claim about additional forecasting information or better decisions.

Shared references do not always add positive interaction. If forecast, outcome and baseline all equal `U`, all three raw ranks move together, but both contrasts are exactly zero. This is why “the same low references raise both ranks” is only part of the explanation: baseline subtraction and each record's map-specific values matter.

From the package root, run `python examples/reference_rank/reference_rank_example.py` to reproduce the table and expectations using exact rational arithmetic. The script also checks complete cancellation and an example with negative reference interaction. It requires only Python's standard library.
