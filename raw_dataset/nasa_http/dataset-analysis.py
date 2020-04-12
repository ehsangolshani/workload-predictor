import pandas as pd


def analyze(source_csv_path: str):
    df = pd.read_csv(source_csv_path, delimiter=',')
    print('number of all requests: ', len(df))
    print('number of all requests in July: ', len(df[df.month == 'Jul']))
    print('number of all requests in August: ', len(df[df.month == 'Aug']))

    df = df[['source_host', 'seconds_from_start']]
    df_source_host = df['source_host']
    unique_source_hosts = df_source_host.unique()
    print('number of source hosts: ', len(unique_source_hosts))

    counter = 0
    sum = 0

    for i, source in enumerate(unique_source_hosts):
        if i % 10000 == 1:
            print(sum / counter, '  ')

        tmp_df = df[df.source_host == source]
        seconds_from_start = tmp_df['seconds_from_start'].iloc[1:]
        previous_seconds_from_start = tmp_df['seconds_from_start'].shift(periods=1).iloc[1:]
        diff = seconds_from_start - previous_seconds_from_start
        sum += diff.sum()
        counter += len(diff)

        # for index, row in tmp_df.iterrows():
        #     seconds_from_start = row['seconds_from_start']
        #     if last_row_seconds_from_start is not None:
        #         diff = seconds_from_start - last_row_seconds_from_start
        #         print(diff)
        #         sum += diff
        #         counter += 1
        #     last_row_seconds_from_start = seconds_from_start

    print('average time between two rewuests from same source host: ', sum / counter)

    print()


if __name__ == '__main__':
    analyze(source_csv_path='NASA_access_log_analysis.csv')

    print('finish')
