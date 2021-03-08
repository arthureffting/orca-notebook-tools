import datetime
from fuzzywuzzy import process
import pandas as pd
import matplotlib.pyplot as plt


def range_to_str(start, end):
    """
    :return: Something like "aug0293_oct1093" based on start and end range.
    """
    month_array = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    start_day = str(start.day)
    start_day = start_day if start.day >= 10 else "0" + start_day
    start_month = month_array[start.month - 1]
    start_year = str(start.year)[2:]
    end_day = str(end.day)
    end_day = end_day if end.day >= 10 else "0" + end_day
    end_month = month_array[end.month - 1]
    end_year = str(end.year)[2:]
    return start_month + start_day + start_year + "_" + end_month + end_day + end_year


def str_to_range(range_as_string):
    start_string = range_as_string.split("_")[0]
    end_string = range_as_string.split("_")[1]

    start_month = (start_string[0:3])
    start_day = int(start_string[3:5])
    start_year_end = int(start_string[5:7])

    end_month = (end_string[0:3])
    end_day = int(end_string[3:5])
    end_year_end = int(end_string[5:7])

    month_map = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }

    if start_month not in month_map or end_month not in month_map:
        raise Exception("Invalid month mapping: " + range_as_string)

    start_month = month_map[start_month]
    end_month = month_map[end_month]
    start = datetime.datetime(start_year_end + (2000 if start_year_end < 10 else 1900),
                              start_month,
                              start_day).date()
    end = datetime.datetime(end_year_end + (2000 if start_year_end < 10 else 1900),
                            end_month,
                            end_day).date()

    return start, end


def match(text, str2Match):
    ratio = process.extract(str2Match, [text])
    return ratio[0][1]


def contains(text, str2Match, probability=0.5):
    prob = match(text, str2Match) / 100
    return prob >= probability


def plot(x_axis,
         data,
         title="Orca plot",
         resample=None,
         tick_by="year",
         save_to=None,
         fillna=True):
    df = pd.DataFrame()
    dt_dates = pd.to_datetime(x_axis)
    df['Date'] = dt_dates
    df.set_index('Date', inplace=True)

    for key in data:
        df[key] = data[key]

    if tick_by == "month":
        min_year = min(x_axis)
        max_year = max(x_axis)
        current = min_year
        ticks = [min(x_axis)]
        while current < max_year:
            current = current + datetime.timedelta(days=30)
            ticks.append(current)
    else:
        min_year = min(x_axis, key=lambda x: x.year).year
        max_year = max(x_axis, key=lambda x: x.year).year
        ticks = [datetime.datetime(i, 1, 1) for i in range(min_year, max_year + 2)]

    if resample is not None:
        df = df.resample(resample).mean()

    if fillna:
        df.fillna(0, inplace=True)

    ax = df.plot(title=title,
                 kind="line",
                 stacked=False,
                 xticks=ticks,
                 colormap='tab20',
                 lw=1.0)
    if save_to:
        plt.savefig(save_to)
    plt.show()
