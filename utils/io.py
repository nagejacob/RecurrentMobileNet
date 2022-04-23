import datetime

def date_time():
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
    return date_time

def log(log_file, str, also_print=True):
    with open(log_file, 'a+') as F:
        F.write(str)
    if also_print:
        print(str, end='')