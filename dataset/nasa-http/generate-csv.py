import csv
from typing import List


def generate_scv(path: str):
    with open(file=path, mode="rb") as file:
        line_count = 0
        with open(file=path + '.csv', mode='w') as csv_file:
            fieldnames = ['year', 'month', 'day', 'hour', 'minute', 'second',
                          'seconds_from_start', 'request_type', 'reply_code']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_writer = csv.writer(csv_file)
            for line in file:
                line = str(line)
                raw_fields = line.split()
                if len(raw_fields) != 10:
                    continue
                # TODO (EhsanGolshani): change calculation of seconds_from_start value,
                #  it should be started from first event, not from first of the day
                fields = extract_fields(raw_fields=raw_fields)
                csv_writer.writerow(fields)
                line_count += 1
                if line_count % 100000 == 0:
                    print(line_count)


def extract_fields(raw_fields: List[str]) -> [int, str, int, int, int, int, int, str, str]:
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

    seconds_from_start = ((day - 1) * 24 * 60 * 60) + (hour * 60 * 60) + (minute * 60) + second

    request_type = raw_fields[5]
    request_type = request_type[1:]
    reply_code = raw_fields[8]

    return year, month, day, hour, minute, second, seconds_from_start, request_type, reply_code


if __name__ == '__main__':
    generate_scv("NASA_access_log_Aug95")
    generate_scv("NASA_access_log_Jul95")
