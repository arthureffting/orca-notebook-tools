from analysis.src.data import Data
from analysis.src.utils import plot

data = Data.read("timed_results.json")

# Gets all the mentions for a specific tape
tape_mentions = data.with_tape_mentions(tapes=["322B"])

# TODO: THIS needs to be discussed.
# It is very hard to identify when a tape starts or ends.
# Even with correct handwriting recognition.
# Using sam
start = tape_mentions.get_lines()[0]
end = tape_mentions.get_lines()[1]

# Uses start and end of tape to get all the data inbetween
data = data.between_lines(start, end)

# Gets all unique pods mentioned in the timespan of the tape
pods = data.unique_pods()
# Filtering only pod/individuals starting with A
pods = [pod for pod in pods if pod[0] == 'A']

pod_data = []

for pod in pods:
    time, a_mentions = data.get_axis(
        lambda t: 0 if len(t) == 0 else len(t.get_pod_mentions(pods=[pod])))
    pod_data.append({
        "pod": pod,
        "values": a_mentions,
        "total": sum(a_mentions)
    })

# Get the 5 most cited pods/individuals only
pod_data.sort(key=lambda x: x["total"])
pod_data = pod_data[-5:]

plot_data = {}
for p in pod_data:
    plot_data[p["pod"]] = p["values"]

# Plot results
plot(time,
     plot_data,
     title="322B pods",
     save_to="examples/tape_ranges.png",
     fillna=True,
     resample="3M")

# Save all lines inbetween the tape
data.save("examples/322B_pod_mentions.json")
