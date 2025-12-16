import pandas as pd
import numpy as np
import os

def get_hurricane_data(filepath):
    """
    Parse HURDAT2 format file into a clean pandas DataFrame.
    
    HURDAT2 has two types of lines:
    - Header: AL/EP code, Name, Number of entries
    - Data: Date, Time, Record, Status, Lat, Lon, MaxWind, MinPressure, ...
    
    Args:
        filepath: Path to HURDAT2 .txt file
        
    Returns:
        pandas DataFrame with all storm observations
    """
    
    storms = []
    current_storm = None
    counter = 0
    with open(filepath, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split(',')]

            # Header line: starts with AL or EP (basin code)
            if parts[0].startswith(('AL', 'EP')):
                current_storm = {
                    'storm_id': parts[0],
                    'name': parts[1] if parts[1] != 'UNNAMED' else np.nan,
                    'num_entries': int(parts[2])
                }
            
            # Data line: date in format YYYYMMDD
            else:
                try:
                    date = parts[0]
                    time = parts[1]
                    
                    # Parse latitude/longitude (e.g., "28.0N" -> 28.0)
                    lat = float(parts[4][:-1])
                    if parts[4][-1] == 'S':
                        lat = -lat
                    
                    lon = float(parts[5][:-1])
                    if parts[5][-1] == 'W':
                        lon = -lon
                    
                    # Build observation record
                    obs = {
                        'storm_id': current_storm['storm_id'],
                        'storm_name': current_storm['name'],
                        'date': date,
                        'time': time,
                        'year': int(date[:4]),
                        'record_id': parts[2],
                        'status': parts[3],  # TS, HU, TD, etc.
                        'latitude': lat,
                        'longitude': lon,
                        'max_wind_kt': int(parts[6]) if parts[6] and parts[6] != '-999' else np.nan,
                        'min_pressure_mb': int(parts[7]) if parts[7] and parts[7] != '-999' else np.nan,
                    }
                    
                    storms.append(obs)
                    
                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    continue
    
    df = pd.DataFrame(storms)
    
    # Convert date/time to datetime
    df['datetime'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M')
    
    return df


# Example usage
if __name__ == "__main__":
    # Parse the file
    txt_folder = 'Data'
    txt_name = 'hurdat2-1851-2023-051124.txt'

    txt_filepath = os.path.join(txt_folder, txt_name)

    if os.path.isfile(txt_filepath):
        df = get_hurricane_data(txt_filepath)
    else:
        raise ValueError(f'File not found! Looking for: {txt_filepath}')

    # Save to CSV
    json_name = os.path.join(txt_folder, 'hurdat2_all_events.csv')
    df.to_csv(json_name, index=False)
    
    print(f"Success! HURDAT2 downloaded here: {json_name}.")
    print("\nKey fields in output:")
    print("  - storm_id: Unique identifier (e.g., AL012005)")
    print("  - storm_name: Hurricane name")
    print("  - year: Year of observation")
    print("  - record_id: Record Identifier (Reference: https://www.aoml.noaa.gov/hrd/hurdat/hurdat2-format.pdf)")
    print("  - status: Storm classification (Reference: https://www.aoml.noaa.gov/hrd/hurdat/hurdat2-format.pdf)")
    print("  - max_wind_kt: Maximum sustained wind in knots")
    print("  - latitude, longitude: Position")