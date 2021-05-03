from analysis.src.data import Data
from analysis.src.utils import plot

data = Data.read("timed_results.json")

# Gets the mentions for different pods and divides
# by the total length of the time range (in pages)
# so as to get a relative frequency of mentions
time, a_mentions = data.get_axis(lambda t: 0 if len(t) == 0 else len(t.get_pod_mentions(pods=["A"])) / t.page_count())
_, t_mentions = data.get_axis(lambda t: 0 if len(t) == 0 else len(t.get_pod_mentions(pods=["T"])) / t.page_count())
data = {
    "A": a_mentions,
    "T": t_mentions,
}

# Plot results
plot(time,
     data,
     title="A vs T comparison",
     save_to="examples/orca_mentions.png",
     fillna=True,
     resample="6M")
