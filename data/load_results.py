import json
import os
import statistics

from src.data import Data
from src.page_detail import PageDetail
from src.time_range import TimeRange


def load(years=None):
    processed = []
    year_data_filenames = os.listdir("data")
    time_ranges = {}
    if years is not None:
        year_data_filenames = [p for p in year_data_filenames if any([p.startswith(y) for y in years])]
    else:
        year_data_filenames = [p for p in year_data_filenames]
    year_data_filenames.sort()
    for year_data_filename in year_data_filenames:
        print("[Loading]", year_data_filename)
        with open(os.path.join("data", year_data_filename)) as year_data:
            data = json.load(year_data)
            for page_data in data:

                if page_data["filename"] in processed:
                    continue
                else:
                    processed.append(page_data["filename"])

                # Split each image in two sides
                xs = [line["start_position"][0] for line in page_data["lines"]]
                std_dev = statistics.stdev(xs)
                mean_x = sum(xs) / len(xs)

                if std_dev < 180:
                    # Single page, dont split
                    pages = [PageDetail({
                        "year": page_data["year"],
                        "filename": page_data["filename"],
                        "lines": page_data["lines"]
                    })]
                else:
                    pages = [PageDetail({
                        "year": page_data["year"],
                        "filename": page_data["filename"],
                        "lines": [line for line in page_data["lines"] if line["start_position"][0] < mean_x]
                    }), PageDetail({
                        "year": page_data["year"],
                        "filename": page_data["filename"],
                        "lines": [line for line in page_data["lines"] if line["start_position"][0] > mean_x]
                    })]

                for page in pages:
                    if page.range_index not in time_ranges:
                        time_ranges[page.range_index] = TimeRange(page.range_index)
                    time_ranges[page.range_index].add_page(page)
    print("[Total files]", len(processed))
    all_time_ranges = []
    for range_index in time_ranges:
        for time_range in time_ranges[range_index].split():
            all_time_ranges.append(time_range)
    return Data(all_time_ranges)