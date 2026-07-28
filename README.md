# Forecasting and Profiling Crime Patterns Across U.S. States (2010 to 2024)

DATA 975 Capstone project. This repository contains the analysis notebook, the
trained model, and a small Flask web application that serves predictions from it.

## What this project does

It combines three public government datasets into one state by year panel and
models the violent crime rate from socioeconomic conditions and arrest patterns.

| Source | What it provides | Level |
| --- | --- | --- |
| FBI Crime Data Explorer, estimated crimes | Violent and property crime counts, population | State and year |
| U.S. Census ACS 5 year | Median income, poverty rate, Gini index, median age | State and year |
| FBI arrests by state | Arrest counts by offense type | State and year |

## Results

Models were evaluated using a time-aware train/test split (2010–2021 for training and 2022–2024 for testing) after removing data leakage.

| Model | Train R² | Test R² | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| **Random Forest** | **0.971** | **0.397** | **83.118** | **131.416** |
| Gradient Boosting | 0.898 | 0.341 | 90.002 | 137.298 |
| Linear Regression | 0.433 | 0.216 | 123.883 | 149.805 |
| Decision Tree | 1.000 | -0.087 | 104.040 | 176.416 |

The Random Forest model achieved the best performance on the held-out test data and was selected as the final model for deployment.

## Repository layout

```
.
├── notebooks/
│   └── Edona_CapstoneProject2.ipynb    analysis and modeling
├── model/
│   └── violent_crime_model.joblib      trained model plus feature list
├── templates/
│   └── index.html                      web form
├── app.py                              Flask application
├── save_model_from_colab.py            cell used to export the model
├── requirements.txt
└── README.md
```

## Running the web app locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

There is also a JSON endpoint:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"gini_index": 0.47, "poverty_rate": 14.2, "median_income": 62000, "median_age": 38.5, "years_since_2010": 12}'
```

## Reproducing the analysis

The notebook was written in Google Colab and reads its input files from
`/content/`. To rerun it, upload the two source CSV files to the Colab session
and supply your own Census API key as a Colab secret named `CENSUS_KEY`.

Get a free key at https://api.census.gov/data/key_signup.html

## Limitations

- Results are associations, not causes. A tree model on observational data
  cannot show that any factor causes crime.
- Missing arrest values were treated as zero, which assumes no arrests rather
  than no report submitted.
- 2010 ACS values were carried back from 2011, and 2024 values were carried
  forward from 2023, because those vintages were not available.
- This is a course project and is not intended for policy or operational use.

## Author

Edona Halilaj, DATA 975 Capstone, Summer 2026
