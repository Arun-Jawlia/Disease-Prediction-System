| Decision | Implementation |
|---|---|
| Train/test split | 80/20 |
| Split strategy | Stratified |
| Random state | 42 |
| Diabetes invalid zeros | Convert selected zeros to NaN |
| Numerical missing values | Median |
| Categorical missing values | Most frequent |
| Categorical encoding | One-hot |
| Numerical scaling | StandardScaler |
| Liver duplicates | Remove exact duplicates |
| Liver target | `1 → 1`, `2 → 0` |
| Raw datasets | Never modified |
| Leakage prevention | Fit preprocessing only on training data |
| Model preprocessing | Eventually stored inside complete Pipeline |