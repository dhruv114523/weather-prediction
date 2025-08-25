import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def gg_season(data, x_col, y_col, season_col, title=None, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    
    # Get unique seasons/years and create color palette
    unique_seasons = sorted(data[season_col].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_seasons)))
    
    # Plot each season/year as a separate line
    for i, season in enumerate(unique_seasons):
        season_data = data[data[season_col] == season].sort_values(x_col)
        plt.plot(season_data[x_col], season_data[y_col], 
                color=colors[i], label=f'{season}', alpha=0.8, linewidth=2)
    
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.title(title or f'Seasonal Plot: {y_col} by {x_col}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def gg_season_monthly(data, y_col, date_col, title=None, figsize=(12, 8)):
    """
    Seasonal plot showing monthly patterns across years
    Similar to R's gg_season() with monthly aggregation
    """
    # Prepare data
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    
    # Aggregate by year and month
    monthly_data = df.groupby(['year', 'month'])[y_col].mean().reset_index()
    
    plt.figure(figsize=figsize)
    
    # Plot each year as a separate line
    unique_years = sorted(monthly_data['year'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_years)))
    
    for i, year in enumerate(unique_years):
        year_data = monthly_data[monthly_data['year'] == year]
        plt.plot(year_data['month'], year_data[y_col], 
                color=colors[i], label=f'{year}', marker='o', linewidth=2)
    
    plt.xlabel('Month')
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.title(title or f'Seasonal Plot: {y_col} by Month')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

# Load your Seattle weather data
print("Creating seasonal plots for Seattle weather data...")
df = pd.read_csv('seattle-weather.csv')
df['date'] = pd.to_datetime(df['date'])

# Add useful date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear

print("\n1. Temperature Seasonal Plot (by day of year)")
# Plot 1: Temperature by day of year, colored by year
gg_season(df, 'day_of_year', 'temp_max', 'year', 
          'Seattle Max Temperature by Day of Year (2012-2015)')
plt.show()

print("\n2. Temperature Monthly Pattern (by month)")
# Plot 2: Temperature by month, showing each year
gg_season_monthly(df, 'temp_max', 'date', 
                  'Seattle Max Temperature Monthly Pattern')
plt.show()

print("\n3. Precipitation Seasonal Plot")
# Plot 3: Precipitation by day of year
gg_season(df, 'day_of_year', 'precipitation', 'year',
          'Seattle Precipitation by Day of Year (2012-2015)')
plt.show()

print("\n4. Advanced Seasonal Analysis with Subplots")
# Create a more comprehensive seasonal analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Temperature max by day of year
ax1 = axes[0, 0]
unique_years = sorted(df['year'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_years)))
for i, year in enumerate(unique_years):
    year_data = df[df['year'] == year].sort_values('day_of_year')
    ax1.plot(year_data['day_of_year'], year_data['temp_max'], 
             color=colors[i], label=f'{year}', alpha=0.8)
ax1.set_title('Max Temperature by Day of Year')
ax1.set_xlabel('Day of Year')
ax1.set_ylabel('Max Temperature (°C)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Temperature min by day of year
ax2 = axes[0, 1]
for i, year in enumerate(unique_years):
    year_data = df[df['year'] == year].sort_values('day_of_year')
    ax2.plot(year_data['day_of_year'], year_data['temp_min'], 
             color=colors[i], label=f'{year}', alpha=0.8)
ax2.set_title('Min Temperature by Day of Year')
ax2.set_xlabel('Day of Year')
ax2.set_ylabel('Min Temperature (°C)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Precipitation by day of year
ax3 = axes[1, 0]
for i, year in enumerate(unique_years):
    year_data = df[df['year'] == year].sort_values('day_of_year')
    # Use rolling average to smooth precipitation data
    year_data['precip_smooth'] = year_data['precipitation'].rolling(window=7, center=True).mean()
    ax3.plot(year_data['day_of_year'], year_data['precip_smooth'], 
             color=colors[i], label=f'{year}', alpha=0.8)
ax3.set_title('Precipitation by Day of Year (7-day rolling average)')
ax3.set_xlabel('Day of Year')
ax3.set_ylabel('Precipitation (mm)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Wind by day of year
ax4 = axes[1, 1]
for i, year in enumerate(unique_years):
    year_data = df[df['year'] == year].sort_values('day_of_year')
    year_data['wind_smooth'] = year_data['wind'].rolling(window=7, center=True).mean()
    ax4.plot(year_data['day_of_year'], year_data['wind_smooth'], 
             color=colors[i], label=f'{year}', alpha=0.8)
ax4.set_title('Wind Speed by Day of Year (7-day rolling average)')
ax4.set_xlabel('Day of Year')
ax4.set_ylabel('Wind Speed')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n5. Monthly Box Plots (Alternative seasonal view)")
# Monthly box plots - another way to view seasonal patterns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Temperature box plots
df.boxplot(column='temp_max', by='month', ax=axes[0,0])
axes[0,0].set_title('Max Temperature by Month')
axes[0,0].set_xlabel('Month')

df.boxplot(column='temp_min', by='month', ax=axes[0,1])
axes[0,1].set_title('Min Temperature by Month')
axes[0,1].set_xlabel('Month')

df.boxplot(column='precipitation', by='month', ax=axes[1,0])
axes[1,0].set_title('Precipitation by Month')
axes[1,0].set_xlabel('Month')

df.boxplot(column='wind', by='month', ax=axes[1,1])
axes[1,1].set_title('Wind Speed by Month')
axes[1,1].set_xlabel('Month')

plt.tight_layout()
plt.show()

print("\nSeasonal plots complete! These show:")
print("- Clear seasonal patterns in temperature (summer peaks, winter lows)")
print("- Precipitation patterns (wet winters, dry summers)")
print("- Year-to-year variations within seasonal trends")
print("- These patterns justify including date features in your weather prediction model!")
