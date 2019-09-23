import pandas as pd


def preprocess(source_csv_path: str, output_csv_path: str, time_step: int = 60):
    df = pd.read_csv(source_csv_path, delimiter=',')
    df = df[['day', 'hour', 'minute', 'seconds_from_start']]
    req_num_df = pd.DataFrame(columns=['day', 'hour', 'minute', 'request_numbers'])
    print(df)

    req_num = 0

    start_step = df.loc[0]['seconds_from_start']
    next_step = start_step + time_step

    new_df_index = 0
    for i, row in df.iterrows():
        if row['seconds_from_start'] < next_step:
            req_num += 1
        else:
            req_num_df.loc[new_df_index] = [row['day'], row['hour'], row['minute'], req_num]
            req_num = 0
            new_df_index += 1
            start_step = next_step
            next_step = start_step + time_step

            if new_df_index % 10000 == 0:
                print(new_df_index)

    print(req_num_df)
    
    req_num_df.to_csv(output_csv_path, sep=',', index=False)


if __name__ == '__main__':
    # in seconds
    preprocess('NASA_access_log_Aug95.csv', 'nasa_temporal_request_number_dataset_August95_30s.csv', 30)
    preprocess('NASA_access_log_Jul95.csv', 'nasa_temporal_request_number_dataset_July95_30s.csv', 30)
