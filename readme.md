F1 Race Prediction Project

Overview: This project predicts Formula 1 race results using data collected via the FastF1 module. The dataset was engineered to be independent of team names, driver names, track names, and grid size. This ensures future proofing against changes in the sport. For example, with Cadillac joining in 2026 there will be no issues with qualifying or finishing positions. With rookie drivers debuting there will be no schema breakage. With Madrid Ring track in 2026 there will be no dependency on track names. With grid size changes from 20 to 22 cars results remain valid through relative positions.

Goal: The objective is to predict race results using two approaches. First, a ranker model that directly predicts finishing positions. Second, a regression model that predicts race pace, then sorts drivers by predicted pace to derive finishing order.
<<<<<<< HEAD
Methodology: Driver and team performance were aggregated over the last five races. Track structure was categorized into street, hybrid, and classic. Races were grouped by race id for consistency. Long runs were calculated from FP sessions and missing values were filled with the closest reliable substitute. Qualifying and finishing positions were converted to relative values, position divided by grid size, for grid size independence. Categorical features such as weather, setup demand, and track environment were encoded as integers. Continuous features such as track length, performance metrics, long runs, and race pace were standardized with StandardScaler. Models were trained on all races except the last season, and tested on the last season.
Results: Best ranker model achieved prediction accuracy of 7 out of 20 finishing positions. Best regression model achieved prediction accuracy of 7 out of 20 finishing positions via race pace sorting.
Files: f1_encoded_not_scaled.csv is the encoded dataset for ranker models.
f1_encoded_scaled.csv is the encoded and scaled dataset for regression models.
scaler.pkl is the saved StandardScaler for consistent preprocessing of new race data.
=======

Methodology: Driver and team performance were aggregated over the last five races. Track structure was categorized into street, hybrid, and classic. Races were grouped by race id for consistency. Long runs were calculated from FP sessions and missing values were filled with the closest reliable substitute. Qualifying and finishing positions were converted to relative values, position divided by grid size, for grid size independence. Categorical features such as weather, setup demand, and track environment were encoded as integers. Continuous features such as track length, performance metrics, long runs, and race pace were standardized with StandardScaler. Models were trained on all races except the last ten, and tested on the last ten races.

Results: Best ranker model achieved prediction accuracy of x out of 20 finishing positions. Best regression model achieved prediction accuracy of y out of 20 finishing positions via race pace sorting.

>>>>>>> 2b5dae17a40228d5414f5b125a5a0d01f967fcc8
Future Work: Extend feature engineering with tire degradation and pit stop strategies.
Incorporate weather variability beyond binary dry and rainy.
Explore ensemble approaches combining ranker and regression predictions.
