# Seattle Weather Prediction Model

A machine learning project that predicts weather types in Seattle using historical weather data with seasonal patterns.

## 🎯 Project Overview

This project demonstrates how incorporating date features significantly improves weather prediction accuracy. The model uses Random Forest classification to predict weather types (rain, sun, fog, drizzle, snow) based on:
- Weather conditions (precipitation, temperature, wind)
- Date features (month, day of year, seasonal patterns)
- Cyclical encoding for temporal features

## 📊 Results

- **Baseline accuracy** : 83.6%
- **With date features**: 85.3%
- **Improvement**: +1.7 percentage points
- **Overfitting**: Well-controlled (3.2% gap)

## 🔧 Features

- **Weather Features**: precipitation, temp_max, temp_min, wind
- **Date Features**: month, day_of_year, season
- **Cyclical Encoding**: sin/cos transformations for temporal cycles
- **Seasonal Analysis**: Comprehensive seasonal pattern visualization

## 📁 Files

- `weather_predictor.py` - Main prediction model with overfitting controls
- `seasonal_plots.py` - Python equivalent of R's gg_season() for seasonal visualization
- `seattle-weather.csv` - Historical weather data (2012-2015)
- `weather_model.pkl` - Trained model for predictions

## 🚀 Quick Start

```python
from weather_predictor import predict_weather

# Example prediction
result = predict_weather(
    precipitation=0.0,
    temp_max=25,
    temp_min=15,
    wind=3.0,
    month=7,
    day_of_year=196
)

print(f"Prediction: {result['predicted_weather']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 📈 Key Insights

1. **Date features are crucial** for weather prediction
2. **Seasonal patterns** strongly influence weather types
3. **Cyclical encoding** helps model understand temporal relationships
4. **Summer**: 68% sunny, 22% rain
5. **Winter**: 59% rain, 24% sunny

## 🛠️ Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 🎓 Educational Value

This project demonstrates:
- Feature engineering for time series data
- Overfitting reduction techniques
- Seasonal pattern analysis
- Model evaluation and comparison
- Production-ready ML pipeline

## 📊 Model Performance

```
              precision    recall  f1-score   support
     rain         0.98      0.93      0.95       129
     sun          0.77      0.98      0.86       128
     fog          0.00      0.00      0.00        20
   drizzle        1.00      0.18      0.31        11
     snow         1.00      0.40      0.57         5

    accuracy                          0.85       293
```

## 🔄 Cyclical Features

The model uses sine and cosine transformations to handle cyclical nature of time:
- January and December are close in feature space
- Smooth seasonal transitions
- Better generalization across years
