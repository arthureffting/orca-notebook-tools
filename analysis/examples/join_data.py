from analysis.src.data import Data
from analysis.src.utils import plot
import datetime

data = Data.read("timed_results.json")

data = data.subset(datetime.datetime(1990, 1, 1), datetime.datetime(2000, 1, 1))

# Joins data containing "A1" with data containing "calls" to get
# a set of lines which contains both
calls = data.with_keyword_mentions("calls", prob_threshold=0.75)

a = data.with_pod_mentions(pods=["A"])
a_calls = Data.join(a, calls)

n = data.with_pod_mentions(pods=["N"])
n_calls = Data.join(n, calls)

# Since both joins produce data with different sets of time ranges,
# we need to equalize them to make sure they operate on the same timespan
a_calls, n_calls = Data.equalize(a_calls, n_calls)

# Plot results
# We can divide the number of lines
# by the number of pages to see how
# frequently these mentions occurred.
time, a_calls_mentions = a_calls.get_axis(lambda t: t.line_count() / t.page_count())
_, n_calls_mentions = n_calls.get_axis(lambda t: t.line_count() / t.page_count())

data = {
    "A calls": a_calls_mentions,
    "N calls": n_calls_mentions,
}
plot(time,
     data,
     title="A1 call frequency",
     fillna=True,
     save_to="examples/join_data.png",
     tick_by="year",
     resample="3M")
