# Predicting power plant output from ambient conditions

A regression project on the UCI Combined Cycle Power Plant dataset. The idea: a plant's electrical output
depends on the weather at the air intake. Hot thin air gives the turbine less to work with, cool dense air
gives it more. So how accurately can you predict the net hourly output from four ambient readings?

The data is 9,568 hourly records taken over six years while the plant ran at full load. Four inputs
(ambient temperature, exhaust vacuum, ambient pressure, relative humidity) and one target, net hourly
electrical output in MW.

## What's in here

- `analysis.ipynb` is the full walkthrough with code, charts and commentary.
- `figures/` has the saved plots.
- `data/ccpp.csv` is the dataset (converted from the original Excel file).

## Approach

Nothing exotic, and that is partly the point. I looked at the correlations first, confirmed temperature is
the dominant driver and that it is negative (which the air-density physics predicts), then compared a
linear regression baseline against a random forest and gradient boosting. Features scaled for the linear
model, 80/20 train/test split.

## Results

On the held-out test set:

| model | RMSE (MW) | MAE (MW) | R2 |
|---|---|---|---|
| linear regression | 4.50 | 3.60 | 0.930 |
| random forest | 3.23 | 2.31 | 0.964 |
| gradient boosting | 3.83 | 2.95 | 0.950 |

The random forest was best, landing about 2 to 3 MW off on average on an output that ranges from roughly
420 to 496 MW, so under 1% error. Temperature and exhaust vacuum carry almost all the predictive weight.

The tree models beat the linear baseline, but only by a little, because the main relationship really is
close to linear. That is worth saying out loud: a simple model already explains most of the variance here,
so the complex one is not buying much.

I also checked the two plots that actually matter for a regression model, predicted versus actual (points
sit tight on the diagonal) and residuals (a flat band around zero with no leftover pattern). Both look the
way you want, which means the model is not systematically missing anything.

## Limitations

The plant is always at full load in this data, so none of this speaks to part-load behaviour. And a model
trained on six years of one plant will not transfer to a different plant without retraining. For a real
deployment I would also watch for sensor drift rather than trusting a calibration from years ago.

## Running it

```
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

Data source: Combined Cycle Power Plant dataset, UCI Machine Learning Repository.
