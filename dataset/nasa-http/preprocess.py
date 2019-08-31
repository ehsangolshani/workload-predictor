import pandas as pd

if __name__ == '__main__':

    # in seconds
    time_step = 30
    df = pd.read_csv('NASA_access_log_Aug95.csv', delimiter=',')
    df = df[['day', 'hour', 'minute', 'seconds_from_start']]
    req_num_df = pd.DataFrame(columns=['day', 'hour', 'minute', 'request_numbers'])
    print(df)

    req_num = 0

    start_step = df.loc[0]['seconds_from_start']
    next_step = start_step + time_step

    for i, row in df.iterrows():
        if row['seconds_from_start'] < next_step:
            req_num += 1
        else:
            req_num_df = req_num_df.append(
                {'day': row['day'],
                 'hour': row['hour'],
                 'minute': row['minute'],
                 'request_numbers': req_num
                 }, ignore_index=True)
            req_num = 0
            start_step = next_step
            next_step = start_step + time_step

        if i > 10000:
            break

    print(req_num_df)
    req_num_df.to_csv("nasa_temporal_request_number_dataset.csv", sep=',')
