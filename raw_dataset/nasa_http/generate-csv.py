import csv
from typing import List


def generate_scv(paths: List[str], output_path):
    with open(file=output_path, mode='w') as csv_file:
        fieldnames = ['source_host', 'year', 'month', 'day', 'hour', 'minute', 'second',
                      'seconds_from_start', 'request_type', 'reply_code']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_writer = csv.writer(csv_file)
        line_count = 0
        last_second_from_start = 0
        for path in paths:
            last_second_from_start_in_previous_month = last_second_from_start
            with open(file=path, mode="rb") as file:
                for line in file:
                    line = str(line)
                    raw_fields = line.split()
                    if len(raw_fields) != 10:
                        continue
                    # TODO (EhsanGolshani): change calculation of seconds_from_start value,
                    #  it should be started from first event, not from first of the day
                    fields = extract_fields(raw_fields=raw_fields)
                    last_second_from_start = fields[6]
                    if fields[1] == 'Aug':
                        fields[6] += last_second_from_start_in_previous_month
                    csv_writer.writerow(fields)
                    line_count += 1
                    if line_count % 100000 == 0:
                        print(line_count)


def extract_fields(raw_fields: List[str]) -> [str, int, str, int, int, int, int, int, str, str]:
    source_host = raw_fields[0][2:]
    time = raw_fields[3]
    time = time[1:]
    time_splitted = time.split(sep=':')
    hour = int(time_splitted[1])
    minute = int(time_splitted[2])
    second = int(time_splitted[3])

    date = time_splitted[0]
    date_splitted = date.split(sep='/')
    day = int(date_splitted[0])
    month = date_splitted[1]
    year = int(date_splitted[2])

    seconds_from_start_in_this_month = ((day - 1) * 24 * 60 * 60) + (hour * 60 * 60) + (minute * 60) + second

    request_type = raw_fields[5]
    request_type = request_type[1:]
    reply_code = raw_fields[8]

    return [source_host, year, month, day, hour, minute, second,
            seconds_from_start_in_this_month, request_type, reply_code]


if __name__ == '__main__':
    generate_scv(paths=['NASA_access_log_Jul95', 'NASA_access_log_Aug95'],
                 output_path='NASA_access_log_analysis.csv')
