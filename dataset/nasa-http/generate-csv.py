import csv
from typing import List
from datetime import datetime


def generate_scv(path: str):
    with open(file=path, mode="rb") as file:
        line_count = 0
        with open(file=path + '.csv', mode='w') as csv_file:
            fieldnames = ['date', 'hour', 'minutes', 'seconds', 'request_type', 'reply_code']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_writer = csv.writer(csv_file)
            for line in file:
                line = str(line)
                raw_fields = line.split()
                if len(raw_fields) != 10:
                    continue
                fields = extract_fields(raw_fields=raw_fields)
                csv_writer.writerow(fields)
                line_count += 1
                if line_count % 100000 == 0:
                    print(line_count)


def extract_fields(raw_fields: List[str]) -> [str, str, str, str, str, str]:
    time = raw_fields[3]
    time = time[1:]
    time_splitted = time.split(sep=':')
    date = time_splitted[0]
    hour = time_splitted[1]
    minute = time_splitted[2]
    second = time_splitted[3]
    request_type = raw_fields[5]
    request_type = request_type[1:]
    reply_code = raw_fields[8]

    return date, hour, minute, second, request_type, reply_code


if __name__ == '__main__':
    generate_scv("NASA_access_log_Aug95")
    generate_scv("NASA_access_log_Jul95")
