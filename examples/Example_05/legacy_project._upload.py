from heat_battery.simulations.postgresql_project import Project
from heat_battery.simulations.jobs import job_from_legacy_folder
import os
import pandas as pd

project = Project(
    'project_legacy',
    if_exists='override',
    )

dirs = os.listdir('examples/Example_05/data_backups/results_size_5_v4')
#dirs += os.listdir('examples/Example_05/data_backups/results_backup_v3')
print(dirs)

jobs = []
for dir in dirs:
    dir_path = os.path.join('examples/Example_05/data_backups/results_size_5_v4', dir)
    job = job_from_legacy_folder(dir_path)
    print(job)
    jobs.append(job)

project.add_jobs(jobs)
df_jobs = project.get_jobs(as_dataframe=True)

columns = None
for dir in dirs:
    print(dir)
    dir_path = os.path.join('examples/Example_05/data_backups/results_size_5_v4', dir)
    res_df = pd.read_csv(os.path.join(dir_path, 'unsteady.csv'))
    res_df['Heating_on'] = res_df['Heating_on'].astype(int)
    print(res_df.head())
    try:
        res_df.drop(columns=['t_remain_avg'], inplace=True)
    except:
        pass
    if columns is None:
        columns = res_df.columns
    else:
        if (set(columns) != set(res_df.columns)):
            print(set(columns))
            print(set(res_df.columns))
            print(f"Columns do not match for dir: {dir}")
    signature = dir.split('_')[-1][:20]
    project.upload_result_table(signature, res_df)

res_names = project.get_result_table_names()
print(res_names)

# jobs = project.get_jobs(as_dataframe=True)  
# print(jobs[['signature','probe_columns']])

# import time
# signature_to_read = '307f326af7ab107af0a9'
# start_time = time.time()
# df = project._get_result_dataframe_direct(signature_to_read)
# print(f"Time taken read by direct read: {time.time() - start_time:.4f} seconds")
# print(df.shape)          
# print(df.head())          
