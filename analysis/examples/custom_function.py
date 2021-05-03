from analysis.src.data import Data
from analysis.src.utils import plot
import datetime

data = Data.read("../timed_results.json")
data = data.subset(datetime.datetime(1990, 1, 1), datetime.datetime(2004, 1, 1))

# Function that counts the lines within a time_range that contain a certain keyword
# The "contains" method of line details might take a while.
def keyword_frequency(time_range, keyword, prob_threshold=0.9):
    lines_containing = [line for line in time_range.get_lines() if line.contains(keyword, prob_threshold)]
    return len(lines_containing)


# Gets the mentions for different pods and divides
# by the total length of the time range (in pages)
# so as to get a relative frequency of mentions
time, call_mentions = data.get_axis(lambda t: keyword_frequency(t, "calls"))
data = {
    "calls": call_mentions,
}

# Plot results
plot(time,
     data,
     title="Calls mentions",
     fillna=True,
     save_to="custom_function.png",
     tick_by="year",
     resample="3M")
