F1 Race Prediction Project

Overview: This project predicts Formula 1 race results using data collected via the FastF1 module. The dataset was engineered to be independent of team names, driver names, track names, and grid size. This ensures future proofing against changes in the sport. For example, with Cadillac joining in 2026 there will be no issues with qualifying or finishing positions. With rookie drivers debuting there will be no schema breakage. With Madrid Ring track in 2026 there will be no dependency on track names. With grid size changes from 20 to 22 cars results remain valid through relative positions.

Goal: The objective is to predict race results using two approaches. First, a ranker model that directly predicts finishing positions. Second, a regression model that predicts race pace, then sorts drivers by predicted pace to derive finishing order.

Methodology: Driver and team performance were aggregated over the last five races. Track structure was categorized into street, hybrid, and classic. Races were grouped by race id for consistency. Long runs were calculated from FP sessions and missing values were filled with the closest reliable substitute. Qualifying and finishing positions were converted to relative values, position divided by grid size, for grid size independence. Categorical features such as weather, setup demand, and track environment were encoded as integers. Continuous features such as track length, performance metrics, long runs, and race pace were standardized with StandardScaler. Models were trained on all races except the last ten, and tested on the last ten races.

Results: Best ranker model achieved prediction accuracy of x out of 20 finishing positions. Best regression model achieved prediction accuracy of y out of 20 finishing positions via race pace sorting.


F1 is a sport that heavily depends on strategy, luck, safety car deployment, and driver skills. You really can’t predict it with perfection — a driver with a bad car and poor results might suddenly shine after an upgrade, while another with strong pace can be undone by a bad strategy call, like McLaren in Qatar 2025.
There is room for improvement in modeling and analytics: better features, smarter algorithms, more context about weather, track types, and team upgrades can all push accuracy higher. But there is no perfection — the chaos factor will always remain. The best we can do is get closer, reduce the blind spots, and accept that unpredictability is part of what makes F1 thrilling

Future Work:Improving the models, Extend feature engineering with tire degradation and pit stop strategies.
Incorporate weather variability beyond binary dry and rainy.
Explore ensemble approaches combining ranker and regression predictions.
