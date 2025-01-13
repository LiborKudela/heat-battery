from .fetchers import pvgis
from ..utilities import hash_data, load_data_binary, save_data_binary
import inspect
import os
from mpi4py import MPI
from ..config import get_config_item

#TODO check if pv database name for sellected locations is smart or not
# locations (latitute[°], longitude[°], altitude[ft])
locations = {
    'Brno-FME': (49.22465761983221, 16.574647060135998, 966), # Faculty of mechanical engineering in Brno, Czech republic
}
#TODO: add renewable.ninja fetcher (can estimate hourly heating/cooling demand)

class CachedMeteoDataLoader:
    def __init__(self, cache_dir='wheather'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.temp_air_day_mean_methods = {
            'CZ_mean': lambda x: (x.iloc[7] + x.iloc[14] + 2*x.iloc[21])/4
        }

    def fetch_hourly(self,
        location,
        usehorizon=True,
        userhorizon=None,
        pvcalculation=True,
        temp_air_day_mean_func='CZ_mean',
        heating_season_start_month=9,
        heating_season_end_month=5,
        peakpower=1,
        mountingplace='building',
        loss=0,
        trackingtype=0,
        optimal_surface_tilt=False,
        optimalangles=False,
        outputformat='csv',
        map_variables=True,
        timeout=30,
        ):

        """This function fetches data for photovoltaic system placed in a 'location'
        with maximum time-span given in the 'pvgis' database"""

        spec = inspect.getfullargspec(self.fetch_hourly).args
        local_scope = locals()
        call_data = dict(zip(spec, [eval(arg, local_scope) for arg in spec]))
        call_data.pop('self')
        call_data_hash = hash_data(call_data)
        file_path = os.path.join(self.cache_dir, call_data_hash)
        data_path = file_path + ".pvhourly"

        pv_hourly = None
        if MPI.COMM_WORLD.rank == 0 and os.path.isfile(data_path):
            print("Loading hourly data from chache..")
            pv_hourly = load_data_binary(data_path)
        elif MPI.COMM_WORLD.rank == 0:
            print("Loading hourly data from PVGIS..")
            pv_hourly = pvgis.get_pvgis_hourly(
                latitude=location[0],
                longitude=location[1],
                usehorizon=usehorizon,
                userhorizon=userhorizon,
                pvcalculation=pvcalculation,
                peakpower=peakpower,
                mountingplace=mountingplace,
                loss=loss,
                trackingtype=trackingtype,
                optimal_surface_tilt=optimal_surface_tilt,
                optimalangles=optimalangles,
                outputformat=outputformat,
                map_variables=map_variables,
                timeout=timeout,
                )
            df = pv_hourly[0]
            df['elapsed_time'] = (df.index - df.index[0]).total_seconds()
            taamf = self.temp_air_day_mean_methods.get(temp_air_day_mean_func, temp_air_day_mean_func)    
            df['temp_air_day_mean'] = df.resample('D')['temp_air'].transform(taamf)
            df['heating_season'] = True
            df.loc[(df.index.month > heating_season_end_month) & (df.index.month < heating_season_start_month), 'heating_season'] = False
            
            df['heating_pause'] = False
            counter = 0
            for frame in df.resample('D')['temp_air_day_mean']:
                if frame[1].iloc[0] > 13.0:
                    counter = min(counter+1, 3)
                else:
                    counter = max(counter-1, 0)

                if counter == 3:
                    pause = True
                elif counter == 0:
                    pause = False

                df.loc[frame[1].index, 'heating_pause'] = pause

            start_date = pv_hourly[0].index[0]
            end_data = pv_hourly[0].index[-1]
            print(f"Hourly data span: {start_date} - {end_data}")
            print(f"Data columns: {pv_hourly[0].columns}")

            save_data_binary(data_path, pv_hourly)

        pv_hourly = MPI.COMM_WORLD.bcast(pv_hourly)
        return pv_hourly

    def fetch_tmy(self,
        location,
        outputformat='csv',
        usehorizon=True,
        userhorizon=None,
        startyear=None,
        endyear=None,
        map_variables=True,
        timeout=30,
        ):

        """This function fetches typical meteorological year (TMY) for
        photovoltaic system placed in a 'location' with maximum time-span 
        given in the 'pvgis' database"""

        spec = inspect.getfullargspec(self.fetch_tmy).args
        local_scope = locals()
        call_data = dict(zip(spec, [eval(arg, local_scope) for arg in spec]))
        call_data.pop('self')
        call_data_hash = hash_data(call_data)
        file_path = os.path.join(self.cache_dir, call_data_hash)
        data_path = file_path + ".pvtmy"

        pv_tmy = None
        if MPI.COMM_WORLD.rank == 0 and os.path.isfile(data_path):
            print("Loading TMY data from chache..")
            pv_tmy = load_data_binary(data_path)
        elif MPI.COMM_WORLD.rank == 0:
            print("Loading TMY data from PVGIS..")
            pv_tmy = pvgis.get_pvgis_tmy(
                latitude=location[0],
                longitude=location[1],
                outputformat=outputformat, 
                usehorizon=usehorizon, 
                userhorizon=userhorizon, 
                startyear=startyear, 
                endyear=endyear, 
                map_variables=map_variables, 
                timeout=timeout,
                )
            save_data_binary(data_path, pv_tmy)

        pv_tmy = MPI.COMM_WORLD.bcast(pv_tmy)
        return pv_tmy

def to_function(df, 
    value_name, offset_date=None, interpolate=True):
    """Return O(1) callable that provides f(t) interface to data
    t - is elapsed time in seconds."""

    if offset_date is not None:
        #TODO: IS this efficient for large datasets?
        offset_time = df[df.index >= offset_date]['elapsed_time'].iloc[0]
    else:
        offset_time = 0.0

    time = df['elapsed_time'].to_numpy()
    val = df[value_name].to_numpy()
    max_i = len(time) - 2
    if interpolate:
        def callable(t):
            t = (t+offset_time)
            i = min(int(t/3600), max_i)
            dv = val[i+1] - val[i]
            res = val[i] + (dv/3600)*(t-time[i])
            return res
        return callable
    else:
        def callable(t):
            t = (t+offset_time)
            i = min(int(t/3600), max_i)
            return val[i]
        return callable