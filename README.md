`Please note: all example scripts' imports are realtive to the project root directory`

# Orcalab notebooks analysis tools

This repository contains a set of python tools to navigate and interpret the orca logbook data.

It operates on handwritten data transcriptions stored as `json` objects in formats described in the `Format`section, and
can be reused if subsequent enhancements to the handwriting recognition of the logbook data are available.

## Structure

Each page has been divided into left and right side, to account for timestamps identified in different sides of the
image. These pages and the lines included in them have then been organized in the following data structures.

### Breakpoints

Since temporal information is already correctly represented in the name of the images of the dataset, a fixed time range
in which each piece of text finds itself is already given. Beyond that, dates encountered in the recognized handwritten
texts are used to create breakpoints , which are used to split these defined initial time ranges into smaller, more
precise ranges.

### Time ranges

By using the breakpoints to split the data, pages and lines have been organized into separate `TimeRange` objects,
according to dates identified in the handwritten text.

### Data

The `Data` class is basically a wrapper for a set of `TimeRange` objects on which operations can be performed. These
operations, in turn, return other instances of `Data`, on which subsequent operations can be performed, also in
combination with other `Data` instances.

The `Data` class also offers functions to make plotting and saving filtered/combined data more easily.

## Usage

Not all available functions are described here, a set of more comprehensive usage examples can be found
under `/examples`.

### Load data

Given a `json` file containing time ranges in the format described in the `Format` section of this document, it can be
loaded onto a `Data` object.

```python
data = Data.read("timed_results.json")
```

### Filter

Some functions are provided to filter the data according to some parameter, for example, filtering lines that contain
specific pod/individual/family mentions.

```python
# By refering only to the family ("A"), all mentions starting with the letter will be considered (e.g A26, A32, A5)
# Optionally, you can also refer to specific individuals (e.g "A26")
a_data = data.with_pod_mentions(pods=["A"])
```

Data can also be filtered by specific keywords according. Levenshtein distance is used to measure similarity and a
threshold for the occurrence probability can be passed to the function.

```python
calls = data.with_keyword_mentions("calls", prob_threshold=0.75)
```

A function is also available to filter out lines containing tape mentions.

```python
tape_322B_mentions = data.with_tape_mentions(tapes=["322B"])
```

### Join

For getting lines that match more than one selection criteria, data can be joined together. For example, we can get
lines that mention `A` and `N` pods together.

```python
# By refering only to the family ("A"), all mentions starting with the letter will be considered (e.g A26, A32, A5)
# Optionally, you can also refer to specific individuals (e.g "A26")
a_data = data.with_pod_mentions(pods=["A"])
n_data = data.with_pod_mentions(pods=["N"])
a_and_n_data = Data.join(a_data, n_data)
```

Or we could find only lines that mention both a specific individual and "calls":

```python
# By refering only to the family ("A"), all mentions starting with the letter will be considered (e.g A26, A32, A5)
# Optionally, you can also refer to specific individuals (e.g "A26")
a26_data = data.with_pod_mentions(pods=["A26"])
calls = data.with_keyword_mentions("calls", prob_threshold=0.75)
a26_calls = Data.join(a26_data, calls)
```

### Data between lines

Since one of the main interests is to evaluate handwritten data of specific tape recordings, we can also use two
specific lines of the dataset to extract only data between them. (e.g the line where a tape starts and the line where a
tape ends). Since this function also returns a `Data` object, all other functions available can then be applied to it (
e.g analyze pod/keyword information for a specific tape).

```python
tape_mentions = data.with_tape_mentions(tapes=["322B"])
start = tape_mentions.get_lines()[0]
end = tape_mentions.get_lines()[1]
data = data.between_lines(start, end)
pods = data.unique_pods() 
```

### Plotting

A function `get_axis` to get x and y axis for plotting data is also provided. This makes it easy to visualize
pod/keyword/tape information quickly. It creates the x axis (time) according to the time ranges within the data
automatically, and generates the y axis data according to a lambda function passed to it that operates on the time
ranges of the data.

```python
# The function passed to the get_axis function gets the count all A pod mentions and divides 
# it by the total pages contained in each time range, giving us a relative frequency of occurrence.
x, y = data.get_axis(lambda t: len(t.get_pod_mentions(pods=["A"])) / t.page_count())

# The keys in the data dict dictate the labels of the plot 
data = {
    "A family mentions": y,
}
# Plot results
# The resample parameter can be used to group together results.(e.g by 6 months)
plot(x,
     data,
     title="My orca plot",
     fillna=True,
     save_to="my_orca_plot.png",
     resample="6M")

# Calling plt.savefig will enable saving the generated plots
```

### Saving

Filtered data can then be saved to `json` files that contain only the relevant data.

```python
# Save all lines inbetween the tape
# The data could then be directly loaded afterwards using Data.load("322B_tape_mentions.json")
tape_322B_mentions = data.with_tape_mentions(tapes=["322B"])
data.save("322B_tape_mentions.json")
```

# Format

## Handwritten data

This tools expect as an entry point a collection of `json` objects containing basic line information for the images
processed. This data is then converted into separate time ranges, which can be loaded in order to be manipulated.

```json
[
  {
    "lines": [
      {
        "line": "0",
        "text": "10.08.93",
        "start_position": [
          450.1103630065918,
          50.72771072387695
        ]
      }
    ]
  }
]
```

## Time range data format

The set of images and lines provided is then converted into a format already containing metadata about the handwritten
text. A list of pod/invididuals identified in the text, as well as whether tape information has been extracted is given
in the metadata.

```json
[
  {
    "range": "jul0183_jan0984",
    "pages": [
      {
        "lines": [
          {
            "year": "1983",
            "filename": "jul0183_nov2284_pg001.png",
            "index": "1",
            "text": "A1 calls. #312B 100%",
            "pod_mentions": [
              "A1"
            ],
            "tape": {
              "number": "312",
              "side": "B",
              "starts": false,
              "ends": true
            }
          }
        ]
      }
    ]
  }
]
```