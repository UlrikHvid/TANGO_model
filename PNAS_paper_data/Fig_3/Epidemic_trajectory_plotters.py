import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d
from datetime import date, timedelta
import pandas as pd
from CaseDataPlotters import *

# Define smoothing function
def smooth(data, window_size):
    return uniform_filter1d(data, size=window_size, mode='nearest')

# Function to compute means and standard deviations
def compute_mean_std(data_arr, Tmax_plot,durations = None):
    mean_arr = np.zeros(Tmax_plot)
    std_arr = np.zeros(Tmax_plot)
    for t in range(Tmax_plot):
        if durations is not None:
            active_indices = durations > t
            if np.any(active_indices):
                mean_arr[t] = np.mean(data_arr[active_indices, t])
                std_arr[t]  = np.std(data_arr[active_indices, t])
        else:
            mean_arr[t] = np.mean(data_arr[:, t])
            std_arr[t]  = np.std(data_arr[:, t])
    return mean_arr, std_arr

def compute_median_quantiles(data_arr, Tmax_plot, durations=None, quantiles=(0.25, 0.75)):
    median_arr = np.zeros(Tmax_plot)
    q_low_arr = np.zeros(Tmax_plot)
    q_high_arr = np.zeros(Tmax_plot)

    # Convert quantiles to percent
    q_perc = [q * 100 for q in quantiles]

    for t in range(Tmax_plot):
        # Select active samples if durations provided
        if durations is not None:
            active_indices = durations > t
            if np.any(active_indices):
                values = data_arr[active_indices, t]
            else:
                values = np.array([])
        else:
            values = data_arr[:, t]

        if values.size > 0:
            median_arr[t] = np.nanmedian(values)
            q_low_arr[t], q_high_arr[t] = np.nanpercentile(values, q_perc)
        else:
            median_arr[t] = np.nan
            q_low_arr[t] = np.nan
            q_high_arr[t] = np.nan

    return median_arr, q_low_arr, q_high_arr


# Function to compute lower and upper bounds, capping the lower bound at zero
def compute_bounds(mean_arr, std_arr):
    lower = np.maximum(mean_arr - std_arr, 0)
    upper = mean_arr + std_arr
    return lower, upper

# Function to smooth data arrays
def smooth_data(mean_arr, lower_arr, upper_arr, window_size):
    mean_smooth = smooth(mean_arr, window_size)
    lower_smooth = smooth(lower_arr, window_size)
    upper_smooth = smooth(upper_arr, window_size)
    return mean_smooth, lower_smooth, upper_smooth

# Plotting functions for each subplot
def plot_IE(dates, mean_IE_smooth, lower_IE_smooth, upper_IE_smooth, xlim_start, xlim_end, factor, date_format,yscale = "linear"):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, mean_IE_smooth / factor, label='Mean')
    plt.fill_between(dates, lower_IE_smooth / factor, upper_IE_smooth / factor, color='b', alpha=0.2, label='Mean ± 1 Std Dev')
    plt.xlabel('Time')
    plt.ylabel("Currently infected")
    plt.title('$I+E$')
    plt.xlim(xlim_start, xlim_end)
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.legend()
    plt.tight_layout()
    plt.yscale(yscale)
    plt.show()

def plot_R(dates, mean_R_smooth, lower_R_smooth, upper_R_smooth, xlim_start, xlim_end, factor, date_format, per_million,yscale = "linear"):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, mean_R_smooth / factor, label='Mean', color='orange')
    plt.fill_between(dates, lower_R_smooth / factor, upper_R_smooth / factor, color='orange', alpha=0.2, label='Mean ± 1 Std Dev')
    plt.xlabel('Time')
    plt.ylabel(r"Cumulative cases / $10^6$" if per_million else "Cumulative cases")
    plt.title('$R$')
    plt.xlim(xlim_start, xlim_end)
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.legend()
    plt.tight_layout()
    plt.yscale(yscale)
    plt.show()

def plot_Reff(dates, mean_Reff_smooth, lower_Reff_smooth, upper_Reff_smooth, xlim_start, xlim_end, date_format,yscale = "linear"):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, mean_Reff_smooth, label='Mean', color='green')
    plt.fill_between(dates, lower_Reff_smooth, upper_Reff_smooth, color='green', alpha=0.2, label='Mean ± 1 Std Dev')
    plt.xlabel('Time')
    plt.ylabel(r'$R_{eff}$')
    plt.title(r'$R_{eff}$')
    plt.xlim(xlim_start, xlim_end)
    plt.axhline(1, ls="dotted", color="k", alpha=0.5)
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.legend()
    plt.tight_layout()
    plt.yscale(yscale)
    plt.show()


#OBS IN THE CODE BELOW: There is a "per capita" boolean option which applies to only the empirical data, but simulation data must be input as either per capita or not

def PlotWeeklyCasesWithSimulation(data, countries, end_date, per_capita, start_date=None, 
                                  special_country=None, yscale="linear", 
                                  ylim=0, title=None, smooth=False, interpolation="quadratic", 
                                  labelsize=15, titlesize=15,legendsize=13,xticksize=13,yticksize=13,rotation = 45, simulation_dates=None, 
                                  mean_IE_smooth=None, lower_IE_smooth=None, upper_IE_smooth=None, lower2_IE_smooth=None, upper2_IE_smooth=None, 
                                  factor=1, linewidth=1.5, Tvac=None, Nvac=None, durVac=None,figsize=(12, 6)):
    fig, ax1 = plt.subplots(figsize=figsize)  # Use ax1 for the primary axis

    if title is None:
        title = f'Weekly Cases up to {end_date.strftime("%Y-%m-%d")}'
    
    earliest_date = pd.Timestamp.max
    all_weekly_cases = []
    
    for country_name in countries:
        country_data = data[(data['country'] == country_name) & (data['date_type'] == 'Onset')].copy()
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
                ax1.fill_between(x_new, y_smooth, color='grey', alpha=0.5, label=country_name + ' data')
            else:
                ax1.plot(x_new, y_smooth, linestyle='-', label=country_name + ' data',linewidth=linewidth,color = "#3d348b")
        else:
            if country_name == special_country:
                ax1.fill_between(weekly_cases['reference_date'], y_data, color='grey', alpha=0.5, label=country_name + ' data')
            else:
                ax1.plot(weekly_cases['reference_date'], y_data, linestyle='-', label=country_name + ' data',linewidth=linewidth,color = "#3d348b")

    # Plot simulation data
    if simulation_dates is not None and mean_IE_smooth is not None:
        ax1.plot(simulation_dates, mean_IE_smooth / factor, label=r'Simulation mean', color='#f35b04',linewidth=linewidth)
    if lower2_IE_smooth is not None:
        ax1.fill_between(simulation_dates, lower2_IE_smooth / factor, upper2_IE_smooth / factor, color="#f35b04", alpha = 0.4, label=r'90% CI')
    if lower_IE_smooth is not None:
        ax1.fill_between(simulation_dates, lower_IE_smooth / factor, upper_IE_smooth / factor, color="#f35b04", alpha = 0.7, label=r'50% CI')
    
    ax1.set_xlabel('Date of symptom onset', fontsize=labelsize)
    ax1.set_yscale(yscale)
    ax1.set_ylabel('Cases per million' if per_capita else 'Weekly Cases', fontsize=labelsize)
    ax1.set_title(title, fontsize=titlesize)

    # Vaccination data (if provided)
    if Tvac is not None and Nvac is not None and durVac is not None:
        Tvac = pd.Timestamp(Tvac)
        # Create a range of dates starting from earliest_date to end_date
        all_dates = pd.date_range(start=earliest_date, end=end_date, freq='D')
        
        # Create an array of zeros until Tvac and then cumulative vaccinations after Tvac
        vac_numbers = np.zeros(len(all_dates))
        vac_start_index = (all_dates >= Tvac).argmax()  # Find index where Tvac starts
        vac_numbers[vac_start_index:vac_start_index + durVac] = Nvac * np.arange(1, durVac + 1)
        vac_numbers[vac_start_index + durVac:] = vac_numbers.max()
        
        # If the vaccination period extends beyond the end date, cap it at the end of the plot
        vac_numbers = np.clip(vac_numbers, 0, Nvac * durVac)
        
        ax2 = ax1.twinx()  # Create secondary y-axis for vaccinations
        ax2.fill_between(all_dates, vac_numbers, color='black',alpha = 0.1, label='Fully immunized')  # Use solid black line
        ax2.set_ylabel('Cumulative Vaccinations per Million', fontsize=labelsize)
        ax2.set_ylim(0,vac_numbers.max()*1.2)
        ax2.legend(loc='upper right',fontsize=legendsize)

    combined_weekly_cases = pd.concat(all_weekly_cases).drop_duplicates().sort_values(by='reference_date')
    unique_dates = combined_weekly_cases['reference_date'].drop_duplicates().reset_index(drop=True)
    xticks = unique_dates[::2]
    ax1.set_xticks(xticks)
    if xticksize:
        ax1.set_xticklabels(xticks)
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    # Set xticks to display only the 1st of the month
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))  # Locator for the 1st of each month
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))  # Formatter to display day and month
    ax1.tick_params(axis='x', rotation=rotation, labelsize=xticksize)  # Adjust tick parameters (rotation and size)
    ax1.tick_params(axis='y', labelsize=yticksize)  # Adjust tick parameters (rotation and size)


    ax1.set_ylim(ylim)
    ax1.set_xlim(earliest_date, end_date)
    if legendsize:
        ax1.legend(loc = 'upper left',fontsize=legendsize)
    ax1.set_axisbelow(True) # Sets the grid underneath data
    ax1.grid()
    fig.tight_layout()
    plt.show()

def PlotCumulativeCasesWithSimulation(data, countries, end_date, per_capita, start_date=None, cum0 = 0, 
                                      special_country=None, ylim=None, yscale="linear",
                                      smooth=False, interpolation='linear', labelsize=15, 
                                      legendsize=15, yticksize=12, y1multiple = 10, y2multiple = 10, xticksize=15, simulation_dates=None, 
                                      mean_R_smooth=None, lower_R_smooth=None, upper_R_smooth=None, lower2_R_smooth=None, upper2_R_smooth=None,
                                      factor=1, linewidth=1.5, Tvac=None, Nvac=None, durVac=None,figsize=(12, 6),rotation = 45):
    fig, ax1 = plt.subplots(figsize=figsize)  # Use ax1 for the primary axis
    
    earliest_date = pd.Timestamp.max
    all_cumulative_cases = []
    
    for country_name in countries:
        country_data = data[(data['country'] == country_name) & (data['date_type'] == 'Onset')].copy()
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
                ax1.fill_between(x_new, y_smooth, color='grey', alpha=0.5, label=country_name + ' data')
            else:
                ax1.plot(x_new, y_smooth, linestyle='-', label=country_name + ' data', linewidth=linewidth, color = "#3d348b")
        else:
            if country_name == special_country:
                ax1.fill_between(cumulative_cases['reference_date'], y_data, color='grey', alpha=0.5, label=country_name + ' data')
            else:
                ax1.plot(cumulative_cases['reference_date'], y_data, linestyle='-', label=country_name + ' data', linewidth=linewidth,color = "#7678ed")
    
    # Plot simulation data
    if simulation_dates is not None and mean_R_smooth is not None:
        ax1.plot(simulation_dates, (mean_R_smooth) / factor + cum0, label=r'Simulation', color='#f35b04', linewidth=linewidth)
    if lower2_R_smooth is not None:
        ax1.fill_between(simulation_dates, (lower2_R_smooth) / factor + cum0, (upper2_R_smooth) / factor + cum0, color='#f35b04', alpha=0.4, 
                         label=r"90% CI")
    if lower_R_smooth is not None:
        ax1.fill_between(simulation_dates, (lower_R_smooth) / factor + cum0, (upper_R_smooth) / factor + cum0, color='#f35b04', alpha=0.7, 
                         label=r"50% CI")
    
    ax1.set_ylim(ylim)
    ax1.set_xlim(earliest_date, end_date)
    ax1.set_yscale(yscale)
    ax1.set_xlabel('Date of symptom onset', fontsize=labelsize, labelpad = 20)
    if per_capita:
        ax1.set_ylabel(r'Cumulative cases / $10^6$', fontsize=labelsize)
    else:
        ax1.set_ylabel('Cumulative cases', fontsize=labelsize)
    
    # Vaccination data (if provided)
    # Note: This creates a twin axis (ax2) on the right for vaccinations
    if Tvac is not None and Nvac is not None and durVac is not None:
        Tvac = pd.Timestamp(Tvac)
        all_dates = pd.date_range(start=earliest_date, end=end_date, freq='D')
        vac_numbers = np.zeros(len(all_dates))
        vac_start_index = (all_dates >= Tvac).argmax()  # Find index where Tvac starts
        vac_numbers[vac_start_index:vac_start_index + durVac] = Nvac * np.arange(1, durVac + 1)
        vac_numbers[vac_start_index + durVac:] = vac_numbers.max()
        
        vac_numbers = np.clip(vac_numbers, 0, Nvac * durVac)
        
        ax2 = ax1.twinx()  # Create secondary y-axis for vaccinations
        ax2.fill_between(all_dates, vac_numbers, color='black', alpha=0.1, label='Fully immunized')
        ax2.set_ylabel('Cumulative Vaccinations per Million', fontsize=labelsize)
        ax2.set_ylim(0, vac_numbers.max() * 1.2)
        ax2.legend(loc='upper right', fontsize=legendsize)

    combined_cumulative_cases = pd.concat(all_cumulative_cases).drop_duplicates().sort_values(by='reference_date')
    unique_dates = combined_cumulative_cases['reference_date'].drop_duplicates().reset_index(drop=True)
    xticks = unique_dates[::2]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    ax1.tick_params(axis='x', rotation=rotation, labelsize=xticksize)
    ax1.tick_params(axis='y', labelsize=yticksize)
    if yscale == 'linear':
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(y1multiple))
    if legendsize:
        ax1.legend(loc='upper left', fontsize=legendsize)
    ax1.set_axisbelow(True) # Sets the grid underneath data
    ax1.grid()
    fig.tight_layout()

    # Add secondary y-axis for MSM percentage if per_capita is True
    if per_capita:
        def left_to_right(x):
            # x = cases per million
            # Fraction of population infected = x / 1,000,000
            # Fraction MSM infected = Fraction pop * (600/5) = Fraction pop * 120
            # Percentage of MSM = Fraction MSM * 100
            # => x * 0.012
            return x * 0.012

        def right_to_left(x):
            # Inverse: from percentage of MSM back to cases per million
            return x / 0.012
        
        ax3 = ax1.secondary_yaxis('right', functions=(left_to_right, right_to_left))
        ax3.set_ylim(left_to_right(ylim[0]), left_to_right(ylim[1]))
        ax3.set_yscale(yscale)
        ax3.set_ylabel('Percentage of MSM', fontsize=labelsize,labelpad = 10)
        ax3.tick_params(axis='y', labelsize=yticksize)
        if yscale == "linear":
            ax3.yaxis.set_major_locator(ticker.MultipleLocator(y2multiple))
    plt.show()

#Generate TI values from sigmoid function
def sigmoid(t, k, t0):
    TSmax = 14
    TSmin = 6.5
    return TSmax - (TSmax - TSmin) / (1 + np.exp(-k * (t - t0)))

def generate_sigmoid_values(start_date, end_date, t0_date, k):
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    t0 = (t0_date - start_date).days

    results = []
    for date in date_range:
        t = (date - start_date).days
        value = sigmoid(t, k, t0) # k must be in units of /days!
        results.append((date.strftime("%Y-%m-%d"), value))

    return results

def generate_sigmoid_series(start_date, end_date, t0_date, k, pad_preweek=True, pad_days=7):
    """
    Generate a mapping from dates to serial-interval values based on a sigmoid curve.

    Parameters:
    - start_date: first date for which the sigmoid is centered (pd.Timestamp or str)
    - end_date:   last date for which to compute values (pd.Timestamp or str)
    - t0_date:    midpoint date of the sigmoid (pd.Timestamp or str)
    - k:          growth rate parameter (/days)
    - pad_preweek: if True, extends the start_date backward by pad_days to cover any "prev_date" lookups
    - pad_days:    number of days to pad before start_date if pad_preweek is True

    Returns:
    - dict mapping each pd.Timestamp between padded_start and end_date to its sigmoid-based TS value
    """
    # Normalize inputs to Timestamps
    orig_start = pd.to_datetime(start_date)
    end_dt     = pd.to_datetime(end_date)
    t0_dt      = pd.to_datetime(t0_date)

    # Optionally pad the start by one week (or custom pad_days)
    if pad_preweek:
        padded_start = orig_start - pd.Timedelta(days=pad_days)
    else:
        padded_start = orig_start

    # Days offset of sigmoid midpoint relative to original start
    t0 = (t0_dt - orig_start).days

    # Create daily index from padded start through end_date
    date_index = pd.date_range(start=padded_start, end=end_dt, freq='D')

    # Build the mapping
    ts_series = {}
    for date in date_index:
        # Compute t relative to original start_date
        t = (date - orig_start).days
        ts_series[date] = sigmoid(t, k, t0)

    return ts_series

def PlotGrowthRateWithSimulation(
    data,countries,end_date,per_capita,ts_series=None,constant_ts=None,start_date=None,special_country=None,yscale="linear",
    ylim=None,title=None,smooth=False,interpolation='linear',labelsize=15,legendsize=15,titlesize=15,xticksize=13,yticksize=13,
    rotation=45,simulation_dates=None,mean_Reff_smooth=None,lower_Reff_smooth=None,upper_Reff_smooth=None, lower2_Reff_smooth=None,upper2_Reff_smooth=None,
    linewidth=1.5, Tvac=None,Nvac=None,durVac=None,figsize=(12, 6)
):
    # Ensure ts input
    if ts_series is None and constant_ts is None:
        raise ValueError("Provide either a ts_series (date->ts mapping) or a constant_ts value.")
    
    fig, ax1 = plt.subplots(figsize=figsize)
    plt.axhline(1, color="k", ls="--", alpha=0.2)

    if title is None:
        title = r'$R_t$ up to {} (backwards calculated)'.format(end_date.strftime("%Y-%m-%d"))

    earliest_date = pd.Timestamp.max
    all_growth_rate_cases = []

    for country_name in countries:
        country_data = data[(data['country'] == country_name) & (data['date_type'] == 'Onset')].copy()
        if country_data.empty:
            print(f"No data available for {country_name}")
            continue
        country_data['reference_date'] = pd.to_datetime(country_data['reference_date'])

        # Filter by start_date
        if start_date:
            prev_date = country_data[country_data['reference_date'] < start_date]['reference_date'].max()
            if pd.notna(prev_date):
                plot_start_date = prev_date + pd.Timedelta(days=7)
                country_data = country_data[country_data['reference_date'] >= prev_date]
            else:
                country_data = country_data[country_data['reference_date'] >= start_date]

        filtered_data = country_data[country_data['reference_date'] <= end_date]
        min_date = filtered_data['reference_date'].min()
        earliest_date = min(earliest_date, min_date)

        weekly_cases = filtered_data.groupby('reference_date')['cases'].sum().reset_index()
        all_growth_rate_cases.append(weekly_cases)

        # Per capita adjustment
        if per_capita:
            pop = population_sizes.get(country_name)
            if not pop:
                print(f"Population size for {country_name} not found.")
                continue
            weekly_cases['cases_per_capita'] = weekly_cases['cases'] / pop * 1_000_000
            y_data = weekly_cases['cases_per_capita']
        else:
            y_data = weekly_cases['cases']

        # Assign ts values
        if ts_series is not None:
            weekly_cases['ts'] = weekly_cases['reference_date'].map(ts_series)
            if weekly_cases['ts'].isnull().any():
                raise ValueError("ts_series missing values for some dates in " + country_name)
        else:
            weekly_cases['ts'] = constant_ts

        # Compute R_t
        weekly_cases['Rt'] = RtFromWeeklyCases(y_data.shift(1), y_data, weekly_cases['ts'])
        valid = weekly_cases.dropna(subset=['Rt'])

        # Plot
        if smooth:
            x_new = pd.date_range(valid['reference_date'].min(), valid['reference_date'].max(), freq='D')
            interp_fn = interp.interp1d(
                valid['reference_date'].map(pd.Timestamp.toordinal),
                valid['Rt'].fillna(1),
                kind=interpolation
            )
            y_smooth = interp_fn(x_new.map(pd.Timestamp.toordinal))
            plot_x, plot_y = x_new, y_smooth
        else:
            plot_x, plot_y = valid['reference_date'], valid['Rt']

        if country_name == special_country:
            ax1.fill_between(plot_x, plot_y, color='grey', alpha=0.5, label=country_name + ' data')
        else:
            ax1.plot(plot_x, plot_y, linestyle='-', label=country_name + ' data', linewidth=linewidth,color = "#3d348b")

    # Simulation overlay
    if simulation_dates is not None and mean_Reff_smooth is not None:
        ax1.plot(simulation_dates, mean_Reff_smooth, label=r'Simulation Reff', linewidth=linewidth, color='#f35b04')
    if lower2_Reff_smooth is not None and upper2_Reff_smooth is not None:
        ax1.fill_between(simulation_dates, lower2_Reff_smooth, upper2_Reff_smooth, alpha=0.4, color='#f35b04', label="90% CI")
    if lower_Reff_smooth is not None and upper_Reff_smooth is not None:
        ax1.fill_between(simulation_dates, lower_Reff_smooth, upper_Reff_smooth, alpha=0.7, color='#f35b04', label="50% CI")

    # Axis formatting
    ax1.set_xlabel('Date of symptom onset', fontsize=labelsize,labelpad = 15)
    ax1.set_yscale(yscale)
    if ylim is not None:
        ax1.set_ylim(ylim)
    ax1.set_xlim(plot_start_date, end_date)
    ax1.set_ylabel(r'$R_t$', fontsize=labelsize)
    ax1.set_title(title, fontsize=titlesize)

    # Vaccination layer
    if Tvac and Nvac and durVac:
        Tvac = pd.Timestamp(Tvac)
        all_dates = pd.date_range(start=earliest_date, end=end_date, freq='D')
        vac = np.zeros(len(all_dates))
        start_idx = (all_dates >= Tvac).argmax()
        vac[start_idx:start_idx + durVac] = Nvac * np.arange(1, durVac + 1)
        vac[start_idx + durVac:] = vac.max()
        vac = np.clip(vac, 0, Nvac * durVac)
        ax2 = ax1.twinx()
        ax2.fill_between(all_dates, vac, alpha=0.1)
        ax2.set_ylabel('Cumulative Vaccinations per Million', fontsize=labelsize)
        ax2.set_ylim(0, vac.max() * 1.2)
        if legendsize:
            ax2.legend(loc='upper right', fontsize=legendsize)

    # X-ticks
    combined = pd.concat(all_growth_rate_cases).drop_duplicates().sort_values('reference_date')
    dates = combined['reference_date'].drop_duplicates().reset_index(drop=True)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    ax1.tick_params(axis='x', rotation=rotation, labelsize=xticksize)
    ax1.tick_params(axis='y', labelsize=yticksize)

    if legendsize:
        ax1.legend(loc='upper left', fontsize=legendsize)
    ax1.set_axisbelow(True) # Sets the grid underneath data
    ax1.grid(True)
    fig.tight_layout()
    plt.show()

def PlotWeeklyCasesWithManyRuns(data, countries, end_date, per_capita, start_date=None, 
                                  special_country=None, yscale="linear", 
                                  ylim=0, title=None, smooth=False, interpolation="quadratic", 
                                  labelsize=15, titlesize=15,legendsize=13,ticksize=13,rotation = 45, simulation_dates=None, 
                                  daily_arr=None, factor=1, linewidth=1.5, Tvac=None, Nvac=None, durVac=None,figsize=(12, 6)):
    fig, ax1 = plt.subplots(figsize=figsize)  # Use ax1 for the primary axis

    if title is None:
        title = f'Weekly Cases up to {end_date.strftime("%Y-%m-%d")}'
    
    earliest_date = pd.Timestamp.max
    all_weekly_cases = []
    
    # Plot simulation data
    if simulation_dates is not None and daily_arr is not None:
        for i,array in enumerate(daily_arr):
            plot_data = smooth(array/factor,window_size = 10)
            if i == 0: # Only one label
                ax1.plot(simulation_dates, plot_data, label=r'Example simulations', color='#f35b04',linewidth=linewidth, alpha = 0.1)
            else:
                ax1.plot(simulation_dates, plot_data, color='#f35b04',linewidth=linewidth, alpha = 0.1)

    for country_name in countries:
        country_data = data[(data['country'] == country_name) & (data['date_type'] == 'Onset')].copy()
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
                ax1.fill_between(x_new, y_smooth, color='grey', alpha=0.5, label=country_name + ' data')
            else:
                ax1.plot(x_new, y_smooth, linestyle='-', label=country_name + ' data',linewidth=linewidth,color = "#3d348b")
        else:
            if country_name == special_country:
                ax1.fill_between(weekly_cases['reference_date'], y_data, color='grey', alpha=0.5, label=country_name + ' data')
            else:
                ax1.plot(weekly_cases['reference_date'], y_data, linestyle='-', label=country_name + ' data',linewidth=linewidth,color = "#7678ed")
    
    ax1.set_xlabel('Date of symptom onset', fontsize=labelsize)
    ax1.set_yscale(yscale)
    ax1.set_ylabel('Cases per million' if per_capita else 'Weekly Cases', fontsize=labelsize)
    ax1.set_title(title, fontsize=titlesize)

    # Vaccination data (if provided)
    if Tvac is not None and Nvac is not None and durVac is not None:
        Tvac = pd.Timestamp(Tvac)
        # Create a range of dates starting from earliest_date to end_date
        all_dates = pd.date_range(start=earliest_date, end=end_date, freq='D')
        
        # Create an array of zeros until Tvac and then cumulative vaccinations after Tvac
        vac_numbers = np.zeros(len(all_dates))
        vac_start_index = (all_dates >= Tvac).argmax()  # Find index where Tvac starts
        vac_numbers[vac_start_index:vac_start_index + durVac] = Nvac * np.arange(1, durVac + 1)
        vac_numbers[vac_start_index + durVac:] = vac_numbers.max()
        
        # If the vaccination period extends beyond the end date, cap it at the end of the plot
        vac_numbers = np.clip(vac_numbers, 0, Nvac * durVac)
        
        ax2 = ax1.twinx()  # Create secondary y-axis for vaccinations
        ax2.fill_between(all_dates, vac_numbers, color='black',alpha = 0.1, label='Fully immunized')  # Use solid black line
        ax2.set_ylabel('Cumulative Vaccinations per Million', fontsize=labelsize)
        ax2.set_ylim(0,vac_numbers.max()*1.2)
        ax2.legend(loc='upper right',fontsize=legendsize)

    combined_weekly_cases = pd.concat(all_weekly_cases).drop_duplicates().sort_values(by='reference_date')
    unique_dates = combined_weekly_cases['reference_date'].drop_duplicates().reset_index(drop=True)
    xticks = unique_dates[::2]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks)
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    # Set xticks to display only the 1st of the month
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))  # Locator for the 1st of each month
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))  # Formatter to display day and month
    ax1.tick_params(axis='x', rotation=rotation, labelsize=ticksize)  # Adjust tick parameters (rotation and size)
    ax1.tick_params(axis='y', labelsize=ticksize)  # Adjust tick parameters (rotation and size)


    ax1.set_ylim(ylim)
    ax1.set_xlim(earliest_date, end_date)
    if legendsize:
        ax1.legend(loc = 'upper left',fontsize=legendsize)
    ax1.set_axisbelow(True) # Sets the grid underneath data
    ax1.grid()
    fig.tight_layout()
    plt.show()