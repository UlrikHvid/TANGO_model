import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from datetime import datetime,date
from scipy.interpolate import CubicSpline, interp1d
import scipy.interpolate as interp
import copy
import os
from cycler import cycler

########## Setup ##########

epidemic_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the current script directory, i.e., epidemic_dir
file_path = os.path.join(epidemic_dir, 'Country_data_by_date_of_symptom_onset.csv')
data = pd.read_csv(file_path)

# Dictionary containing population sizes for different countries
population_sizes = {
    'Luxembourg':           634814,
    'Denmark':              5831404,
    'Finland':              5540720,
    'Ireland':              4982904,
    'Norway':               5421241,
    'Switzerland':          8654622,
    'Austria':              8917205,
    'Hungary':              9749763,
    'Belgium':              11589623,
    'Czechia':              10708981,
    'France':               65273511,
    'Germany':              83240525,
    'Greece':               10423054,
    'Italy':                59554023,
    'Netherlands':          17134872,
    'Poland':               38386000,
    'Portugal':             10196709,
    'Romania':              19237691,
    'Spain':                47351567,
    'Sweden':               10327589,
    'The United Kingdom':   67886011,
    'Europe':               746400000,
}

# Filter the data for European countries
european_countries = data[data['who_region'] == 'EURO']

# Aggregate the cases by reference_date
european_totals = european_countries.groupby('reference_date').agg({'cases': 'sum'}).reset_index()

# Add the country code "Europe" to the aggregated data
european_totals['country'] = 'Europe'
european_totals['iso3'] = 'EUR'
european_totals['who_region'] = 'EURO'
european_totals['who_region_long'] = 'European Region'
european_totals['date_type'] = 'Onset'

# Append the new rows to the original dataframe
data = pd.concat([data, european_totals], ignore_index=True)

EU_jynneos_rollout_date = pd.Timestamp("2022-06-28")
imvanex_approv_date     = pd.Timestamp("2022-07-22")
WHO_declaration_date    = pd.Timestamp("2022-07-23")
ECDC_risk_assessment_date=pd.Timestamp("2022-05-23")

# Sample DataFrame for festival data
data_festival = {
    "Name": [
        "Maspalomas Gay Pride", "Sitges Pride", "CSD Berlin Pride", "Oslo Pride", 
        "Paris Pride", "Dublin LGBTQ+ Pride", "Copenhagen Pride", "Pride in London", "Madrid Pride", 
        "Cologne Pride", "Milan Pride", "Bristol Pride", "Pride in Hull", "Pride Edinburgh", 
        "Budapest Pride", "Pride Barcelona", "Luxembourg Pride", "Toulouse Pride", "Hamburg Pride", 
        "Munich Pride", "Amsterdam Pride", "Brighton Pride", "Zurich Pride Festival", "Stockholm Pride", 
        "Antwerp Pride", "Manchester Pride","Vienna Pride","Mykonos Pride","Brussels Pride","Baltic Pride"
    ],
    "City": [
        "Gran Canaria", "Sitges", "Berlin", "Oslo", 
        "Paris", "Dublin", "Copenhagen", "London", "Madrid", 
        "Cologne", "Milan", "Bristol", "Hull", "Edinburgh", 
        "Budapest", "Barcelona", "Luxembourg", "Toulouse", "Hamburg", 
        "Munich", "Amsterdam", "Brighton", "Zurich", "Stockholm", 
        "Antwerp", "Manchester", "Vienna","Mykonos","Brussels", "Vilnius"
    ],
    "Participants": [
        200000, 30000, 500000, 70000, 
        700000, 60000, 300000, 1500000, 2000000, 
        1200000, 300000, 40000, 50000, 10000, 
        20000, 200000, 10000, 25000, 250000, 
        150000, 500000, 300000, 40000, 50000, 
        90000, 150000, 200000,30000, 120000,20000
    ],
    "Checked": [
        'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes','Yes','Yes','Yes','Yes','Yes'
    ],
    "Start Date": [
        "2022-05-05", "2022-06-08", "2022-07-15", "2022-06-17", 
        "2022-06-25", "2022-06-22", "2022-08-13", "2022-07-02", "2022-07-01", 
        "2022-06-18", "2022-07-02", "2022-07-09", "2022-07-23", "2022-07-02", 
        "2022-07-23", "2022-06-13", "2022-07-01", "2022-07-09", "2022-08-05", 
        "2022-07-02", "2022-07-30", "2022-08-05", "2022-08-19", "2022-08-01", 
        "2022-08-10", "2022-08-26", "2022-06-01", "2022-08-17", "2022-05-20",
        "2022-06-01"
    ],
    "End Date": [
        "2022-05-15", "2022-06-12", "2022-07-23", "2022-06-26", 
        "2022-06-25", "2022-06-26", "2022-08-21", "2022-07-02", "2022-07-10", 
        "2022-07-03", "2022-07-02", "2022-07-09", "2022-07-23", "2022-07-02", 
        "2022-07-23", "2022-06-26", "2022-07-10", "2022-07-09", "2022-08-07", 
        "2022-07-17", "2022-08-07", "2022-08-07", "2022-08-20", "2022-08-07", 
        "2022-08-14", "2022-08-29", "2022-06-12", "2022-08-24", "2022-05-22",
        "2022-06-05"
    ]
}

# Creating the DataFrame
df_festival = pd.DataFrame(data_festival)

# Convert Start Date and End Date to datetime
df_festival['Start Date'] = pd.to_datetime(df_festival['Start Date'])
df_festival['End Date'] = pd.to_datetime(df_festival['End Date'])

# Function to find total number of participants on a given date
def total_participants_on_date(df, query_date):
    query_date = pd.to_datetime(query_date)
    mask = (df['Start Date'] <= query_date) & (df['End Date'] >= query_date)
    total_participants = df.loc[mask, 'Participants'].sum()
    return total_participants

# Generate a date range for the entire year
date_range = pd.date_range(start='2022-05-01', end='2022-09-01')

# Calculate the total participants for each date
total_participants_per_day = []
for single_date in date_range:
    total_participants = total_participants_on_date(df_festival, single_date)
    total_participants_per_day.append(total_participants)

# Create a DataFrame for plotting
plot_data_festival = pd.DataFrame({
    'Date': date_range,
    'Total Participants': total_participants_per_day
})


# Calculate 7-day rolling average
plot_data_festival['Rolling Avg Participants'] = plot_data_festival['Total Participants'].rolling(window=7, center=True).mean()

# Find the first Thursday starting from the first date in the date range
first_thursday = plot_data_festival['Date'][plot_data_festival['Date'].dt.weekday == 3].iloc[0]

# Filter plot_data to include only Thursdays
thursdays_data_festival = plot_data_festival[plot_data_festival['Date'].dt.weekday == 3].copy().reset_index(drop=True)

##################### Plotting functions #####################

########## Visual style ##########
# Define your custom color palette using hex codes
custom_palette = ['#000000', '#3D348B','#7678ED','#F7B801','#F18701','#F35B04' ]

# Set the color palette globally using `cycler`
plt.rcParams['axes.prop_cycle'] = cycler(color=custom_palette)

def RtFromWeeklyCases(I0,I1,ts,dt = 7):
    k = I1/I0
    return (k)**(ts/dt)

def PlotWeeklyCases(data, countries, end_date, per_capita, start_date=None, special_country=None,special_color='grey', 
                    yscale="linear", ylim=None, title=None, smooth=False, interpolation="quadratic", 
                    labelsize=15, legendsize=15, titlesize=0,linewidth=2,xticksize = 14, yticksize = 14,only_onset = True):
    fig = plt.figure(figsize=(12, 6))
    if title is None:
        title = f'Weekly cases up to {end_date.strftime("%Y-%m-%-d")}'
    
    earliest_date = pd.Timestamp.max
    all_weekly_cases = []
    
    for country_name in countries:
        #country_data = data[(data['country'] == country_name) & (data['date_type'] == 'Onset')].copy()
        country_data = data[(data['country'] == country_name)].copy()
        if only_onset:
            country_data = country_data[(data['date_type'] == 'Onset')]
        if country_data.empty:
            print(f"No data available for {country_name}")
            continue
        country_data['reference_date'] = pd.to_datetime(country_data['reference_date'])

        # Apply the start_date filter if provided
        if start_date:
            country_data = country_data[country_data['reference_date'] >= start_date]
        
        filtered_data = country_data[country_data['reference_date'] <= end_date]
        
        min_date = filtered_data['reference_date'].min()
        if min_date < earliest_date:
            earliest_date = min_date
        
        weekly_cases = filtered_data.groupby('reference_date')['cases'].sum().reset_index()
        all_weekly_cases.append(weekly_cases)
        
        if per_capita:
            population_size = population_sizes.get(country_name)
            if not population_size:
                print(f"Population size for {country_name} not found.")
                continue
            weekly_cases['cases_per_capita'] = (weekly_cases['cases'] / population_size) * 1_000_000
            y_data = weekly_cases['cases_per_capita']
        else:
            y_data = weekly_cases['cases']
        
        if smooth:
            x_new = pd.date_range(start=weekly_cases['reference_date'].min(), 
                                  end=weekly_cases['reference_date'].max(), freq='D')
            interpolator = interp.interp1d(weekly_cases['reference_date'].map(pd.Timestamp.toordinal), y_data, kind=interpolation)
            y_smooth = interpolator(x_new.map(pd.Timestamp.toordinal))
            x_new = pd.to_datetime(x_new)
            
            if country_name == special_country:
                #plt.fill_between(x_new, y_smooth, color=special_color, alpha=0.5, label=country_name)
                plt.plot(x_new, y_smooth, color=special_color, label=country_name)
            else:
                plt.plot(x_new, y_smooth, linestyle='-', label=country_name,linewidth=linewidth)
        else:
            if country_name == special_country:
                plt.fill_between(weekly_cases['reference_date'], y_data, color=special_color, alpha=0.5, label=country_name)
            else:
                plt.plot(weekly_cases['reference_date'], y_data, linestyle='-', label=country_name,linewidth=linewidth)
    
    plt.xlabel('Date of symptom onset', fontsize=labelsize)
    plt.yscale(yscale)
    plt.ylabel('Weekly cases per million' if per_capita else 'Weekly cases', fontsize=labelsize)
    if titlesize:
        plt.title(title, fontsize=titlesize)
    
    combined_weekly_cases = pd.concat(all_weekly_cases).drop_duplicates().sort_values(by='reference_date')
    #unique_dates = combined_weekly_cases['reference_date'].drop_duplicates().reset_index(drop=True)
    #xticks = unique_dates[::2]
    #plt.xticks(xticks, rotation=45, ha='right',fontsize = xticksize)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1]))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',fontsize = xticksize)
    plt.yticks(fontsize=yticksize)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim(earliest_date, end_date)
    if legendsize:
        plt.legend(fontsize=legendsize)
    plt.grid()
    fig.tight_layout()
    plt.show()

def PlotCumulativeCases(data, countries, end_date, per_capita, start_date=None, special_country=None,special_color='grey', 
                        ylim=None, smooth=False, interpolation='linear', xlabelsize=15,ylabelsize=15, legendsize=15,linewidth=2,xticksize=14,yticksize=14,yscale="linear",only_onset = True,figsize = (11,6),ymultiple = 25):
    fig = plt.figure(figsize=figsize)
    
    earliest_date = pd.Timestamp.max
    all_cumulative_cases = []
    
    for country_name in countries:
        country_data = data[(data['country'] == country_name)].copy()
        if only_onset:
            country_data = country_data[(data['date_type'] == 'Onset')]
        if country_data.empty:
            print(f"No data available for {country_name}")
            continue
        country_data['reference_date'] = pd.to_datetime(country_data['reference_date'])

        if start_date:
            country_data = country_data[country_data['reference_date'] >= start_date]
        filtered_data = country_data[country_data['reference_date'] <= end_date]
        
        min_date = filtered_data['reference_date'].min()
        if min_date < earliest_date:
            earliest_date = min_date
        
        cumulative_cases = filtered_data.groupby('reference_date')['cases'].sum().cumsum().reset_index()
        cumulative_cases.rename(columns={'cases': 'cumulative_cases'}, inplace=True)
        all_cumulative_cases.append(cumulative_cases)
        
        if per_capita:
            population_size = population_sizes.get(country_name)
            if not population_size:
                print(f"Population size for {country_name} not found.")
                continue
            cumulative_cases['cumulative_cases_per_million'] = cumulative_cases['cumulative_cases'] / population_size * 1e6
            y_data = cumulative_cases['cumulative_cases_per_million']
        else:
            y_data = cumulative_cases['cumulative_cases']
        
        if smooth:
            x_new = pd.date_range(start=cumulative_cases['reference_date'].min(), 
                                  end=cumulative_cases['reference_date'].max(), freq='D')
            interpolator = interp.interp1d(cumulative_cases['reference_date'].map(pd.Timestamp.toordinal), y_data, kind=interpolation)
            y_smooth = interpolator(x_new.map(pd.Timestamp.toordinal))
            x_new = pd.to_datetime(x_new)
            
            if country_name == special_country:
                #plt.fill_between(x_new, y_smooth, color=special_color, alpha=0.5, label=country_name)
                plt.plot(x_new, y_smooth, color=special_color, alpha=1, label=country_name)
            else:
                plt.plot(x_new, y_smooth, linestyle='-', label=country_name, linewidth=linewidth)
        else:
            if country_name == special_country:
                plt.fill_between(cumulative_cases['reference_date'], y_data, color=special_color, alpha=0.5, label=country_name)
            else:
                plt.plot(cumulative_cases['reference_date'], y_data, linestyle='-', label=country_name, linewidth=linewidth)
    
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim(earliest_date, end_date)
    if xlabelsize != 0:
        plt.xlabel('Date of symptom onset', fontsize=xlabelsize)
    if per_capita:
        plt.ylabel('Cumulative cases per million', fontsize=ylabelsize)
    else:
        plt.ylabel('Cumulative cases', fontsize=ylabelsize)
    
    combined_cumulative_cases = pd.concat(all_cumulative_cases).drop_duplicates().sort_values(by='reference_date')
    ax = plt.gca()
    #unique_dates = combined_cumulative_cases['reference_date'].drop_duplicates().reset_index(drop=True)
    #xticks = unique_dates[::2]
    #plt.xticks(xticks, rotation=45, ha='right', fontsize=xticksize)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1]))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',fontsize = xticksize)
    ax.tick_params(axis='y', labelsize=yticksize)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    if legendsize:
        plt.legend(fontsize=legendsize)
    plt.grid()
    plt.yscale(yscale)
    fig.tight_layout()

    # Add secondary y-axis if per_capita is True
    if per_capita:
        # Conversion function from left axis (cases per million) to right axis (percentage of MSM)
        def left_to_right(x):
            # x = cumulative cases per million
            # fraction of MSM infected (as percentage) = x * 0.012
            return x * 0.012

        def right_to_left(x):
            # Inverse function to go back from percentage of MSM to cases per million
            return x / 0.012

        ax_right = ax.secondary_yaxis('right', functions=(left_to_right, right_to_left))
        ax_right.set_ylabel('% of MSM', fontsize=ylabelsize)
        ax_right.tick_params(axis='y', labelsize=yticksize)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ymultiple))
        ax_right.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax_right.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    plt.show()

def PlotGrowthRate(data, countries, end_date, per_capita, ts1, ts2, ts3, tmitig1, tmitig2, 
                   start_date=None, special_country=None,special_color='grey', yscale="linear", ylim=None, 
                   include_festival_data=False, title=None, smooth=False, interpolation='linear', 
                   labelsize=15, legendsize=15, titlesize=15,alpha = 1,linewidth=2,
                   xticksize=14,yticksize=14,only_onset = True,
                   save_csv=False, csv_filename='Rt_data.csv'):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    plt.axhline(1, color="k", ls="--", alpha=0.2)
    if title is None:
        title = r'$R_t$' + f' up to {end_date.strftime("%Y-%m-%-d")}, backwards calculated'
    
    earliest_date = pd.Timestamp.max
    true_earliest_date = pd.Timestamp.max
    all_growth_rate_cases = []
    # CHANGED: prepare list to collect per-country Rt data for CSV
    csv_dfs = []  
    
    for country_name in countries:
        #country_data = data[(data['country'] == country_name) & (data['date_type'] == 'Onset')].copy()
        country_data = data[(data['country'] == country_name)].copy()
        if only_onset:
            country_data = country_data[(data['date_type'] == 'Onset')]
        if country_data.empty:
            print(f"No data available for {country_name}")
            continue
        country_data['reference_date'] = pd.to_datetime(country_data['reference_date'])

        if start_date:
            prev_date = country_data[country_data['reference_date'] < start_date]['reference_date'].max() # Find the first date before or equal to the start_date
            if pd.notna(prev_date):
                plot_start_date = prev_date + pd.Timedelta(days=7)
                if prev_date < true_earliest_date:
                    true_earliest_date = prev_date
                country_data = country_data[country_data['reference_date'] >= prev_date]
            else:
                plot_start_date = start_date
                country_data = country_data[country_data['reference_date'] >= start_date]
        filtered_data = country_data[country_data['reference_date'] <= end_date]
        
        min_date = filtered_data['reference_date'].min()
        if min_date < earliest_date:
            earliest_date = min_date
        
        weekly_cases = filtered_data.groupby('reference_date')['cases'].sum().reset_index()
        all_growth_rate_cases.append(weekly_cases)
        
        if per_capita:
            population_size = population_sizes.get(country_name)
            if not population_size:
                print(f"Population size for {country_name} not found.")
                continue
            weekly_cases['cases_per_capita'] = (weekly_cases['cases'] / population_size) * 1_000_000
            y_data = weekly_cases['cases_per_capita']
        else:
            y_data = weekly_cases['cases']
        
        # Create the ts array based on tmitig1 and tmitig2
        weekly_cases['ts'] = ts1  # Default to ts1
        if tmitig1:
            weekly_cases.loc[weekly_cases['reference_date'] >= tmitig1, 'ts'] = ts2
        if tmitig2:
            weekly_cases.loc[weekly_cases['reference_date'] >= tmitig2, 'ts'] = ts3

        # Calculate growth rate (R_t)
        weekly_cases['Rt'] = RtFromWeeklyCases(y_data.shift(1), y_data, weekly_cases['ts'])

        valid_growth_rate_data = weekly_cases.dropna(subset=['Rt'])

        if smooth:
            x_new = pd.date_range(start=valid_growth_rate_data['reference_date'].min(), 
                                  end=valid_growth_rate_data['reference_date'].max(), freq='D')
            interpolator = interp.interp1d(valid_growth_rate_data['reference_date'].map(pd.Timestamp.toordinal), 
                                           valid_growth_rate_data['Rt'].fillna(1), kind=interpolation)
            y_smooth = interpolator(x_new.map(pd.Timestamp.toordinal))
            x_new = pd.to_datetime(x_new)

            # CHANGED: collect smoothed data for CSV
            df_country = pd.DataFrame({
                'country': country_name,
                'reference_date': x_new,
                'Rt': y_smooth
            })  # CHANGED
            csv_dfs.append(df_country)  # CHANGED

            if country_name == special_country:
                #ax1.fill_between(x_new, y_smooth, color=special_color, alpha=0.5, label=country_name)
                ax1.plot(x_new, y_smooth, color=special_color, alpha=0.5, label=country_name)
            else:
                ax1.plot(x_new, y_smooth, linestyle='-', label=country_name,alpha = alpha,linewidth=linewidth)
        else:
            # CHANGED: collect raw Rt data for CSV
            df_country = valid_growth_rate_data[['reference_date', 'Rt']].copy()
            df_country['country'] = country_name  # CHANGED
            csv_dfs.append(df_country)  # CHANGED

            if country_name == special_country:
                ax1.fill_between(valid_growth_rate_data['reference_date'], valid_growth_rate_data['Rt'], 
                                 color='grey', alpha=0.5, label=country_name)
            else:
                ax1.plot(valid_growth_rate_data['reference_date'], valid_growth_rate_data['Rt'], 
                         linestyle='-', label=country_name, alpha = alpha,linewidth=linewidth)
    
    # CHANGED: after plotting, save CSV if requested
    if save_csv:
        all_csv = pd.concat(csv_dfs).sort_values(by=['country', 'reference_date'])
        all_csv.to_csv(csv_filename, index=False)
        print(f"Saved Rt data to {csv_filename}")  # CHANGED
    
    ax1.set_xlabel('Date of symptom onset', fontsize=labelsize)
    ax1.set_yscale(yscale)
    if ylim is not None:
        ax1.set_ylim(ylim)
    ax1.set_xlim(plot_start_date, end_date)
    ax1.set_ylabel(r'$R_t$', fontsize=labelsize)
    ax1.set_title(title, fontsize=titlesize)
    
    combined_growth_rate_cases = pd.concat(all_growth_rate_cases).drop_duplicates().sort_values(by='reference_date')
    #unique_dates = combined_growth_rate_cases['reference_date'].drop_duplicates().reset_index(drop=True)
    #xticks = unique_dates[1::2]
    ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1]))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right',fontsize = xticksize)
    #ax1.set_xticks(xticks)
    #ax1.set_xticklabels(xticks, rotation=45, ha='right',fontsize = xticksize)
    yticks = np.arange(0,int(valid_growth_rate_data['Rt'].max())+1)
    #print(valid_growth_rate_data)
    ax1.set_yticklabels(yticks,fontsize=yticksize)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    
    if include_festival_data:
        ax2 = ax1.twinx()
        ax2.plot(thursdays_data_festival['Date'], thursdays_data_festival['Rolling Avg Participants'], 
                 marker='o', linestyle='-', color='b', label='Festival Participants (7-day Rolling Avg)')
        ax2.set_ylabel('Total Participants (7-day Rolling Average)')
    
    fig.tight_layout()
    if legendsize:
        ax1.legend(fontsize=legendsize)
    if include_festival_data:
        ax2.legend(loc='upper right')
    ax1.grid(True)
    plt.show()

def FindDateReachingThreshold(data, threshold=10, per_million=False, interpolate=False):
    dates_reaching_threshold = {}
    
    for country in population_sizes.keys():
        country_data = data[(data['country'] == country) & (data['date_type'] == 'Onset')].copy()
        if country_data.empty:
            print(f"No data available for {country}")
            continue
        country_data.loc[:, 'reference_date'] = pd.to_datetime(country_data['reference_date'])
        country_data.sort_values(by='reference_date', inplace=True)
        country_data['cumulative_cases'] = country_data['cases'].cumsum()

        # Adjust threshold based on cases per million, if per_million is True
        adjusted_threshold = threshold
        if per_million:
            if country in population_sizes:
                population = population_sizes[country]
                adjusted_threshold = threshold * (population / 1_000_000)  # Convert to total cases based on per million
            else:
                print(f"Population size for {country} is not available")
                dates_reaching_threshold[country] = None
                continue

        if interpolate:
            # Perform linear interpolation
            cumulative_cases = country_data['cumulative_cases'].values
            dates = country_data['reference_date'].values
            threshold_reached = False

            # Check if the threshold is reached at or before the first date
            if cumulative_cases[0] >= adjusted_threshold:
                dates_reaching_threshold[country] = dates[0].date()
                threshold_reached = True
            else:
                for i in range(len(cumulative_cases) - 1):
                    c1 = cumulative_cases[i]
                    c2 = cumulative_cases[i + 1]
                    t1 = dates[i]
                    t2 = dates[i + 1]
                    if c1 <= adjusted_threshold <= c2:
                        # Avoid division by zero
                        if c2 == c1:
                            interpolated_date = t1
                        else:
                            ratio = (adjusted_threshold - c1) / (c2 - c1)
                            interpolated_date = t1 + (t2 - t1) * ratio
                        dates_reaching_threshold[country] = pd.Timestamp(interpolated_date).date()
                        threshold_reached = True
                        break
            if not threshold_reached:
                dates_reaching_threshold[country] = None
        else:
            # Existing method without interpolation
            date_reached = country_data[country_data['cumulative_cases'] >= adjusted_threshold]['reference_date']
            if not date_reached.empty:
                dates_reaching_threshold[country] = date_reached.iloc[0].date()  # Use .date() to remove time
            else:
                dates_reaching_threshold[country] = None
        
    return dates_reaching_threshold

def FindCumulativeCasesAsOf(target_date,per_million = False):
    cumulative_cases_as_of_date_per_capita = {}
    target_date = pd.to_datetime(target_date)

    for country in population_sizes.keys():
        country_data = data[(data['country'] == country) & (data['date_type'] == 'Onset')].copy()
        if country_data.empty:
            print(f"No data available for {country}")
            continue
        country_data.loc[:, 'reference_date'] = pd.to_datetime(country_data['reference_date'])
        country_data.sort_values(by='reference_date', inplace=True)
        country_data['cumulative_cases'] = country_data['cases'].cumsum()

        # Find the cumulative cases as of the target date
        cases_as_of_date = country_data[country_data['reference_date'] <= target_date]['cumulative_cases'].max()
        if pd.notna(cases_as_of_date):
            population_size = population_sizes.get(country)
            if not population_size:
                print(f"Population size for {country} not found.")
                continue
            if per_million:
                cumulative_cases_as_of_date_per_capita[country] = cases_as_of_date / population_size * 1e6
            else:
                cumulative_cases_as_of_date_per_capita[country] = cases_as_of_date
        else:
            cumulative_cases_as_of_date_per_capita[country] = 0

    return cumulative_cases_as_of_date_per_capita
