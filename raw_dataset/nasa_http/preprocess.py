import pandas as pd


def preprocess(source_csv_path: str, output_csv_path: str, time_step: int = 60, normalize: bool = False):
    df = pd.read_csv(source_csv_path, delimiter=',')
    df = df[['seconds_from_start']]
    req_num_df = pd.DataFrame(columns=['request_rate', 'normalized_request_rate'])

    req_num = 0

    start_step = df.loc[0]['seconds_from_start']
    next_step = start_step + time_step
    following_start_step = next_step
    following_next_step = following_start_step + time_step

    new_df_index = 0
    for i, row in df.iterrows():
        seconds_from_start = row['seconds_from_start']
        if seconds_from_start < next_step:
            req_num += 1
        else:
            while seconds_from_start >= following_next_step:
                req_num_df.loc[new_df_index] = [req_num, 0]
                req_num = 0  # this individual request should not be counted
                new_df_index += 1
                start_step = following_start_step
                next_step = following_next_step
                following_start_step = next_step
                following_next_step = following_start_step + time_step
                if new_df_index % 10000 == 0:
                    print(new_df_index)

            start_step = next_step
            next_step = start_step + time_step
            following_start_step = next_step
            following_next_step = following_start_step + time_step
            req_num_df.loc[new_df_index] = [req_num, 0]
            req_num = 1  # this individual request should be counted too
            new_df_index += 1
            if new_df_index % 10000 == 0:
                print(new_df_index)

    if normalize:
        min_request_rate = req_num_df['request_rate'].min()
        max_request_rate = req_num_df['request_rate'].max()
        counter = 0
        for i, row in req_num_df.iterrows():
            row['normalized_request_rate'] = (row['request_rate'] - min_request_rate) / (
                    max_request_rate - min_request_rate)
            if counter % 10000 == 0:
                print(counter)
            counter += 1

    req_num_df.to_csv(output_csv_path, sep=',', index=False)


if __name__ == '__main__':
    # time_step is in seconds
    preprocess(source_csv_path='NASA_access_log.csv',
               output_csv_path='nasa_temporal_rps_1m.csv',
               time_step=60,
               normalize=True)

    print('created nasa_temporal_rps_1m.csv')

    preprocess(source_csv_path='NASA_access_log.csv',
               output_csv_path='nasa_temporal_rps_5m.csv',
               time_step=300,
               normalize=True)

    print('created nasa_temporal_rps_5m.csv')

    preprocess(source_csv_path='NASA_access_log.csv',
               output_csv_path='nasa_temporal_rps_10m.csv',
               time_step=600,
               normalize=True)

    print('created nasa_temporal_rps_10m.csv')

    preprocess(source_csv_path='NASA_access_log.csv',
               output_csv_path='nasa_temporal_rps_20m.csv',
               time_step=1200,
               normalize=True)

    print('created nasa_temporal_rps_20m.csv')

    preprocess(source_csv_path='NASA_access_log.csv',
               output_csv_path='nasa_temporal_rps_30m.csv',
               time_step=1800,
               normalize=True)

    print('created nasa_temporal_rps_30m.csv')

    preprocess(source_csv_path='NASA_access_log.csv',
               output_csv_path='nasa_temporal_rps_45m.csv',
               time_step=2700,
               normalize=True)

    print('created nasa_temporal_rps_45m.csv')

    preprocess(source_csv_path='NASA_access_log.csv',
               output_csv_path='nasa_temporal_rps_60m.csv',
               time_step=3600,
               normalize=True)

    print('created nasa_temporal_rps_60m.csv')
