from pvlib import solarposition
import pandas as pd

# Define latitude and longitude
latitude = 37.00167   # Replace with your latitude
longitude = 35.32889  # Replace with your longitude

# Calculate solar position
times = pd.date_range(start='2023-11-25', end='2023-11-26', freq='1h', tz='UTC')
solar_pos = solarposition.get_solarposition(times, latitude, longitude)

# Print solar position data
print(solar_pos)

time = pd.Timestamp('2023-11-25 12:00:00', tz='UTC')  # Replace with your desired time
solar_pos = solarposition.get_solarposition(time, latitude, longitude)

solar_elevation = solar_pos.elevation
tilt_angle = 90 - solar_elevation
print(f"tiltangle for the solar panel at {time} is {tilt_angle.values[0]:.2f} degre")
print(f"azumith for the solar panel at {time} is {solar_elevation.values[0]:.2f} degre")