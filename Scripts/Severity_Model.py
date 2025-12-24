from scipy.stats import lognorm, probplot, nbinom
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_hurricane_data

def model_windspeed(df=None):
    
    if df is None:
        json_folder  = 'Data'
        json_name = 'hurdat2_all_events.csv'

        json_filepath = os.path.join(json_folder, json_name)
        df, _ = load_hurricane_data(json_filepath)

    landfall_df = df[df['record_id'] == 'L']

    wind_df = landfall_df.groupby(['year', 'storm_name']).max('max_wind_kt')
    
    n_hurricanes = len(wind_df)
    wind_kt = wind_df['max_wind_kt']

    wind_kt = wind_kt[wind_kt > 0]
    
    wind_mph = wind_kt * 1.15078

    plt.hist(wind_mph, bins=20)
    plt.xlabel("Wind Speed (mph)")
    plt.ylabel("Count")
    plt.title("Observed Hurricane Wind Speeds at Landfall")
    plt.savefig('Results/WindSpeedHistogram', dpi=300, bbox_inches='tight')
    plt.close()
    log_wind = np.log(wind_mph)

    probplot(log_wind, dist="norm", plot=plt)
    plt.title("Q–Q Plot of log(Wind Speed)")
    plt.savefig('Results/WindSpeedProbability', dpi=300, bbox_inches='tight')
    plt.close()

    shape, loc, scale = lognorm.fit(wind_mph, floc=0)
    sigma = shape
    mu = np.log(scale)

    x = np.linspace(min(wind_mph), max(wind_mph), 300)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    plt.hist(wind_mph, bins=30, density=True)
    plt.plot(x, pdf)
    plt.xlabel("Wind Speed (mph)")
    plt.ylabel("Density")
    plt.title("Lognormal Fit to Hurricane Wind Speeds")
    plt.savefig('Results/LognormalFit.png', dpi=300, bbox_inches='tight')
    plt.close()

    return wind_df, shape, loc, scale


if __name__ == "__main__":
    # Parse the file
    json_folder  = 'Data'
    json_name = 'hurdat2_all_events.csv'

    json_filepath = os.path.join(json_folder, json_name)

    df, df_per_year = load_hurricane_data(json_filepath)

    df, _, _, _ = model_windspeed(df)