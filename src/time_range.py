import datetime
import itertools
import numpy as np
from src.breakpoint import Breakpoint
from src.utils import str_to_range, range_to_str


class TimeRange:

    def __init__(self, range_as_string):
        self.index = range_as_string
        self.start, self.end = str_to_range(range_as_string)
        self.pages = []
        pass

    def __len__(self):
        return len(self.pages)

    def add_page(self, page):
        self.pages += [page]

    def get_lines(self):
        lines = []
        for page in self.pages:
            lines += page.lines
        return lines

    def line_count(self):
        return len(self.get_lines())

    def page_count(self):
        return len(self.pages)

    def is_within(self, start, end, allow_uncertainty=True):
        starts_within = self.start >= start.date() and self.start <= end.date()
        ends_within = self.end >= start.date() and self.end <= end.date()
        return starts_within or ends_within if allow_uncertainty else starts_within and ends_within

    def get_axis(self, fn):
        elapsed_days = (self.end - self.start).days
        value = fn(self)
        days = [self.start + datetime.timedelta(days=i) for i in range(elapsed_days)]
        values = [value for i in range(elapsed_days)]
        return days, values

    def get_pod_mentions(self, pods=None):
        pod_mentioning_lines = []
        for page in self.pages:
            for line in page.lines:
                if line.has_pod_mention(pods):
                    pod_mentioning_lines.append(line)
        return pod_mentioning_lines

    def get_breakpoints(self):
        bs = []
        for page in self.pages:
            for line in page.lines:
                if line.has_date():
                    bs += [Breakpoint(self, line)]
        return bs

    def get_valid_breakpoints(self):
        return [b for b in self.get_breakpoints() if b.is_valid()]

    def as_json(self):
        return {
            "range": range_to_str(self.start, self.end),
            "pages": [page.as_json() for page in self.pages],
        }

    def split(self):
        bps = self.get_valid_breakpoint_sequence()
        breakpoint_filenames = [b.line.page for b in bps]
        page_indexes = [0] + [self.pages.index(p) for p in breakpoint_filenames] + [len(self.pages) + 1]
        dates = [str_to_range(self.pages[0].range_index)[0]] + [b.get_date() for b in bps] + [
            str_to_range(self.pages[-1].range_index)[1]]
        ranges = []

        for i in range(len(page_indexes) - 1):
            used_pages = self.pages[page_indexes[i]:page_indexes[i + 1]]
            start, end = dates[i], dates[i + 1]
            set_range_index = range_to_str(start, end)
            new_range = TimeRange(set_range_index)
            for page in used_pages:
                new_range.add_page(page)
            ranges.append(new_range)

        return ranges

    def get_valid_breakpoint_sequence(self):
        """
        Splits a time range in smaller ranges according to the valid breakpoints
        :return: List of subsets of the time range, split by breakpoints
        """
        # To split, it needs to find a proper combination of breakpoints that makes sense,
        bps = self.get_valid_breakpoints()

        for to_remove in range(len(bps)):
            # Check whether the sequence of breakpoints is valid
            if to_remove == 0:
                valid = all([bps[i].get_date() < bps[i + 1].get_date() for i in range(len(bps) - 1)])
                if valid:
                    return bps
            elif to_remove == len(bps) - 1:
                return []
            else:
                available_indexes = range(len(bps))
                combinations = list(itertools.combinations(available_indexes, to_remove))
                for combination in combinations:
                    subset_of_bps = [bp for index, bp in enumerate(bps) if index not in np.asarray(combination)]
                    valid = all([subset_of_bps[i].get_date() < subset_of_bps[i + 1].get_date() for i in
                                 range(len(subset_of_bps) - 1)])
                    if valid:
                        return subset_of_bps
        return bps
