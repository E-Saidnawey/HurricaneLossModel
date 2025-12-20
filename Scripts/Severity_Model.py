from scipy.stats import lognorm, probplot
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def model_windspeed(df):
    
    landfall_df = df[df['record_id'] == 'L']

    wind_df = landfall_df.groupby(['year', 'storm_name']).max('max_wind_kt')

    n_hurricanes = len(wind_df)
    wind_kt = wind_df['max_wind_kt']

    wind_kt = wind_kt[wind_kt > 0]
    
    plt.hist(wind_kt, bins=30)
    plt.xlabel("Wind Speed (mph)")
    plt.ylabel("Count")
    plt.title("Observed Hurricane Wind Speeds at Landfall")
    plt.savefig('Results/WindSpeedHistogram', dpi=300, bbox_inches='tight')
    plt.close()
    log_wind = np.log(wind_kt)

    probplot(log_wind, dist="norm", plot=plt)
    plt.title("Q–Q Plot of log(Wind Speed)")
    plt.savefig('Results/WindSpeedProbability', dpi=300, bbox_inches='tight')
    plt.close()

    shape, loc, scale = lognorm.fit(wind_kt, floc=0)
    sigma = shape
    mu = np.log(scale)
    x = np.linspace(min(wind_kt), max(wind_kt), 300)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    plt.hist(wind_kt, bins=30, density=True)
    plt.plot(x, pdf)
    plt.xlabel("Wind Speed (mph)")
    plt.ylabel("Density")
    plt.title("Lognormal Fit to Hurricane Wind Speeds")
    plt.savefig('Results/LognormalFit.png', dpi=300, bbox_inches='tight')
    plt.close()

    return


def monte_carlo_simulation(statistics, nruns):
    

    return

if __name__ == "__main__":
    # Parse the file
    json_folder  = 'Data'
    json_name = 'hurdat2_all_events.csv'

    json_filepath = os.path.join(json_folder, json_name)

    if os.path.isfile(json_filepath):
        df = pd.read_csv(json_filepath)
        df = df[df['year'] > 1950]
    else:
        raise ValueError(f'File not found! Looking for: {txt_filepath}')

    
    print(f"df loaded successfully: {json_name}.")
    print(df.head())

    df = model_windspeed(df)