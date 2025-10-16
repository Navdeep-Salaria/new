import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Step 1: Load and Preprocess Data
# ---------------------------
def load_and_engineer_features(filepath):
    """Load data and create advanced features"""
    data = pd.read_csv(filepath)
    
    # Feature engineering
    data['temp_humidity_interaction'] = data['temperature'] * data['humidity'] / 100
    data['effective_irradiance'] = data['solar_irradiance'] * (1 - data['cloud_cover'] / 100)
    data['capacity_efficiency'] = data['panel_capacity_kw'] * data['panel_efficiency']
    data['sun_intensity'] = data['solar_irradiance'] * data['sunlight_hours']
    data['performance_ratio'] = data['past_avg_kwh'] / (data['panel_capacity_kw'] * data['sunlight_hours'] + 0.001)
    
    # Temperature efficiency loss (solar panels lose efficiency at high temps)
    data['temp_efficiency_loss'] = np.maximum(0, data['temperature'] - 25) * 0.004
    data['adjusted_efficiency'] = data['panel_efficiency'] * (1 - data['temp_efficiency_loss'])
    
    # Optimal production conditions
    data['optimal_conditions'] = ((data['temperature'] < 35) & 
                                   (data['cloud_cover'] < 30) & 
                                   (data['sunlight_hours'] > 5)).astype(int)
    
    return data

# ---------------------------
# Step 2: Train Model with Cross-Validation
# ---------------------------
def train_optimized_model(data):
    """Train XGBoost with optimized hyperparameters and feature scaling"""
    
    feature_cols = [
        "temperature", "humidity", "cloud_cover", "solar_irradiance",
        "sunlight_hours", "panel_capacity_kw", "panel_efficiency", "past_avg_kwh",
        "temp_humidity_interaction", "effective_irradiance", "capacity_efficiency",
        "sun_intensity", "performance_ratio", "adjusted_efficiency", "optimal_conditions"
    ]
    
    X = data[feature_cols]
    y = data["energy_production_kwh"]
    
    # Handle outliers (cap at 99th percentile)
    for col in X.select_dtypes(include=[np.number]).columns:
        upper_limit = X[col].quantile(0.99)
        lower_limit = X[col].quantile(0.01)
        X[col] = X[col].clip(lower=lower_limit, upper=upper_limit)
    
    # Train-test split with stratification by production range
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling (XGBoost benefits from scaled features in some cases)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optimized XGBoost model
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Faster training
    )
    
    # Train model (compatible with all XGBoost versions)
    model.fit(X_train_scaled, y_train, verbose=False)
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    print("=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"\nTraining Set:")
    print(f"  MAE:  {mean_absolute_error(y_train, y_pred_train):.3f} kWh")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.3f} kWh")
    print(f"  R²:   {r2_score(y_train, y_pred_train):.4f}")
    
    print(f"\nTest Set:")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred_test):.3f} kWh")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f} kWh")
    print(f"  R²:   {r2_score(y_test, y_pred_test):.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                 cv=5, scoring='r2', n_jobs=-1)
    print(f"\n5-Fold CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(feature_importance.head(10).to_string(index=False))
    print("=" * 60)
    
    return model, scaler, feature_cols

# ---------------------------
# Step 3: Get Enhanced Weather Data
# ---------------------------
def get_weather_data(city, api_key):
    """Fetch current and forecast weather data with error handling"""
    try:
        # Current weather
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        clouds = data["clouds"]["all"]
        sunrise = data["sys"]["sunrise"]
        sunset = data["sys"]["sunset"]
        
        # Calculate sunlight hours
        sun_hours = (sunset - sunrise) / 3600
        
        # Improved solar irradiance estimation
        # Base irradiance adjusted for cloud cover and atmospheric conditions
        clear_sky_irradiance = 1000  # W/m² at solar noon
        cloud_factor = (100 - clouds) / 100
        atmospheric_factor = 0.7 + 0.3 * cloud_factor  # Account for atmospheric losses
        solar_irradiance = clear_sky_irradiance * cloud_factor * atmospheric_factor
        
        return {
            'temp': temp,
            'humidity': humidity,
            'clouds': clouds,
            'sun_hours': sun_hours,
            'solar_irradiance': solar_irradiance,
            'city_found': True
        }
    
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error fetching weather data: {e}")
        return {'city_found': False}
    except KeyError as e:
        print(f"\n❌ City not found or invalid API response: {e}")
        return {'city_found': False}

# ---------------------------
# Step 4: Make Prediction with Engineered Features
# ---------------------------
def predict_energy_production(model, scaler, feature_cols, weather_data, 
                               capacity, efficiency, past_avg):
    """Generate prediction with all engineered features"""
    
    # Basic features
    temp = weather_data['temp']
    humidity = weather_data['humidity']
    clouds = weather_data['clouds']
    sun_hours = weather_data['sun_hours']
    solar_irradiance = weather_data['solar_irradiance']
    
    # Engineered features
    temp_humidity_interaction = temp * humidity / 100
    effective_irradiance = solar_irradiance * (1 - clouds / 100)
    capacity_efficiency = capacity * efficiency
    sun_intensity = solar_irradiance * sun_hours
    performance_ratio = past_avg / (capacity * sun_hours + 0.001)
    
    temp_efficiency_loss = max(0, temp - 25) * 0.004
    adjusted_efficiency = efficiency * (1 - temp_efficiency_loss)
    
    optimal_conditions = int((temp < 35) and (clouds < 30) and (sun_hours > 5))
    
    # Create feature array in correct order
    X_today = np.array([[
        temp, humidity, clouds, solar_irradiance, sun_hours,
        capacity, efficiency, past_avg,
        temp_humidity_interaction, effective_irradiance, capacity_efficiency,
        sun_intensity, performance_ratio, adjusted_efficiency, optimal_conditions
    ]])
    
    # Scale and predict
    X_today_scaled = scaler.transform(X_today)
    pred_kwh = model.predict(X_today_scaled)[0]
    
    # Ensure prediction is non-negative and reasonable
    pred_kwh = max(0, pred_kwh)
    max_theoretical = capacity * sun_hours * efficiency
    pred_kwh = min(pred_kwh, max_theoretical * 1.1)  # Cap at 110% of theoretical max
    
    return pred_kwh

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SOLAR ENERGY PRODUCTION PREDICTOR")
    print("=" * 60)
    
    # Load and train model
    print("\n📊 Loading and training model...")
    try:
        data = load_and_engineer_features("solar_data.csv")
        model, scaler, feature_cols = train_optimized_model(data)
    except FileNotFoundError:
        print("❌ Error: solar_data.csv not found!")
        exit(1)
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        exit(1)
    
    # Get user inputs
    print("\n" + "=" * 60)
    print("ENTER YOUR SOLAR PANEL DETAILS")
    print("=" * 60)
    
    city = input("\n🌍 Enter your city name: ").strip()
    
    try:
        capacity = float(input("⚡ Enter solar panel capacity (kW): "))
        efficiency = float(input("📈 Enter panel efficiency (%): ")) / 100
        past_avg = float(input("📊 Enter past average daily production (kWh): "))
        
        if capacity <= 0 or efficiency <= 0 or efficiency > 1 or past_avg < 0:
            print("❌ Invalid input values!")
            exit(1)
            
    except ValueError:
        print("❌ Please enter valid numeric values!")
        exit(1)
    
    # Get weather data
    API_KEY = "16fe821c78b208d560b97536869ba446"
    print(f"\n🌤️  Fetching weather data for {city}...")
    weather_data = get_weather_data(city, API_KEY)
    
    if not weather_data['city_found']:
        print("❌ Unable to fetch weather data. Please check city name and try again.")
        exit(1)
    
    # Make prediction
    pred_kwh = predict_energy_production(
        model, scaler, feature_cols, weather_data,
        capacity, efficiency, past_avg
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n📍 Location: {city}")
    print(f"🌡️  Temperature: {weather_data['temp']:.1f}°C")
    print(f"💧 Humidity: {weather_data['humidity']}%")
    print(f"☁️  Cloud Cover: {weather_data['clouds']}%")
    print(f"☀️  Sunlight Hours: {weather_data['sun_hours']:.2f} hrs")
    print(f"📊 Solar Irradiance: {weather_data['solar_irradiance']:.0f} W/m²")
    print(f"\n⚡ Panel Capacity: {capacity} kW")
    print(f"📈 Panel Efficiency: {efficiency*100:.1f}%")
    print(f"\n" + "=" * 60)
    print(f"🔮 PREDICTED ENERGY PRODUCTION: {pred_kwh:.2f} kWh")
    print("=" * 60)
    
    # Calculate potential earnings (example rate)
    rate_per_kwh = 0.12  # $0.12 per kWh (adjust based on your region)
    potential_earnings = pred_kwh * rate_per_kwh
    print(f"\n💰 Potential earnings today: ${potential_earnings:.2f}")
    print(f"📅 Estimated monthly production: {pred_kwh * 30:.2f} kWh")
    print(f"💵 Estimated monthly earnings: ${potential_earnings * 30:.2f}\n")