import json
from analysis.src.page_detail import PageDetail
from analysis.src.time_range import TimeRange


class Data:

    def __init__(self, time_ranges):
        self.time_ranges = time_ranges

    def get_breakpoints(self):
        bs = []
        for range in self.time_ranges:
            for bp in range.get_breakpoints():
                bs.append(bp)
        return bs

    def get_valid_breakpoints(self):
        return [b for b in self.get_breakpoints() if b.is_valid()]

    def get_lines(self):
        lines = []
        for t in self.time_ranges:
            lines += t.get_lines()
        return lines

    def unique_pods(self):
        pods = []
        for line in self.get_lines():
            if line.has_pod_mention():
                pods += line.pod_mentions
        return list(set(pods))

    def with_tape_mentions(self, tapes=None):
        """
        :param tapes: List of tapes to look for
        :return: Data with ONLY lines containing at least one of the tapes mentioned
        """
        new_time_ranges = []
        for time_range in self.time_ranges:
            kept_pages = []
            for page in time_range.pages:
                page_copy = page.copy_with_tape_mentions(tapes)
                if len(page_copy.lines) > 0:
                    kept_pages.append(page_copy)
            if len(kept_pages):
                new_time_range = TimeRange(time_range.index)
                for page in kept_pages:
                    new_time_range.add_page(page)
                new_time_ranges.append(new_time_range)
        return Data(new_time_ranges)

    def with_keyword_mentions(self, keyword, prob_threshold=0.75):
        new_time_ranges = []
        for time_range in self.time_ranges:
            kept_pages = []
            for page in time_range.pages:
                page_copy = page.copy_with_keyword_mentions(keyword, prob_threshold)
                if len(page_copy.lines) > 0:
                    kept_pages.append(page_copy)
            if len(kept_pages):
                new_time_range = TimeRange(time_range.index)
                for page in kept_pages:
                    new_time_range.add_page(page)
                new_time_ranges.append(new_time_range)
        return Data(new_time_ranges)

    def with_pod_mentions(self, pods=None):
        new_time_ranges = []
        for time_range in self.time_ranges:
            kept_pages = []
            for page in time_range.pages:
                page_copy = page.copy_with_pod_mentions(pods)
                if len(page_copy.lines) > 0:
                    kept_pages.append(page_copy)
            if len(kept_pages):
                new_time_range = TimeRange(time_range.index)
                for page in kept_pages:
                    new_time_range.add_page(page)
                new_time_ranges.append(new_time_range)
        return Data(new_time_ranges)

    def get_axis(self, fn):
        axis = []
        for time_range in [t for t in self.time_ranges if len(t) > 0]:
            this_days, this_values = time_range.get_axis(fn)
            axis += [{
                "day": this_days[i],
                "value": this_values[i]
            } for i in range(len(this_days))]

        axis.sort(key=lambda x: x["day"])
        return [x["day"] for x in axis], [x["value"] for x in axis]

    def subset(self, start, end, allow_uncertain=True):
        """
        :param allow_uncertain: Whether to allow data for which the date is not certainly contained in the range provided.
        :param start: Starting datetime for the subset
        :param end: Ending datetime for the subset
        :return: Subset containing only data within the given range
        """
        return Data([t for t in self.time_ranges if t.is_within(start, end, allow_uncertainty=allow_uncertain)])

    def save(self, destination="timed_results.json"):
        json_data = [d.as_json() for d in self.time_ranges]
        with open(destination, 'w') as f:
            json.dump(json_data, f)

    @staticmethod
    def read(source="timed_results.json"):
        with open(source) as f:
            data = json.load(f)
            time_ranges = []
            for time_range_json in data:
                time_range = TimeRange(time_range_json["range"])
                for page_json in time_range_json["pages"]:
                    page = PageDetail.from_json(page_json)
                    time_range.add_page(page)
                time_ranges.append(time_range)
            return Data(time_ranges)

    @staticmethod
    def equalize(first, second):
        for second_tr in second.time_ranges:
            if first.get_time_range(second_tr.index) is None:
                first.time_ranges.append(TimeRange(second_tr.index))
        for first_tr in first.time_ranges:
            if second.get_time_range(first_tr.index) is None:
                second.time_ranges.append(TimeRange(first_tr.index))
        return first, second

    @staticmethod
    def join(first, second):
        """
        :return: A new set of data containing only lines present in both datasets provided
        """
        # Time ranges that only exist in one are disconsidered
        common_time_ranges = [t.index for t in first.time_ranges if t.index in [t2.index for t2 in second.time_ranges]]

        page_map = {}

        for tr_index in common_time_ranges:
            if tr_index not in page_map:
                page_map[tr_index] = {}
            first_time_range = first.get_time_range(tr_index)
            second_time_range = second.get_time_range(tr_index)
            for first_line in first_time_range.get_lines():
                if first_line.page.filename not in page_map[tr_index]:
                    page_map[tr_index][first_line.page.filename] = PageDetail({
                        "filename": first_line.page.filename,
                        "year": first_line.page.year,
                        "lines": []
                    })
                for second_line in second_time_range.get_lines():
                    if first_line.equals(second_line):
                        page_map[tr_index][first_line.page.filename].lines.append(first_line)

        joined_trs = []
        for tr_index in page_map:
            time_range = TimeRange(tr_index)
            pages = page_map[tr_index]
            for page_filename in pages:
                time_range.add_page(pages[page_filename])
            joined_trs.append(time_range)
        return Data(joined_trs)

    def get_time_range(self, index):
        for t in self.time_ranges:
            if t.index == index:
                return t
        return None

    def print(self):
        json_data = [d.as_json() for d in self.time_ranges]
        indented = json.dumps(json_data, indent=4)
        print(indented)

    def between_lines(self, first_line, second_line):
        new_time_ranges = []
        # Find the lines, record where they were
        first_page_pointer = None
        last_page_pointer = None
        for t in self.time_ranges:
            for page_index, page in enumerate(t.pages):
                if first_page_pointer is None:
                    for line in page.lines:
                        if first_line.equals(line):
                            first_page_pointer = {
                                "tr_index": t.index,
                                "page_index": page_index
                            }
                if last_page_pointer is None:
                    for line in page.lines:
                        if second_line.equals(line):
                            last_page_pointer = {
                                "tr_index": t.index,
                                "page_index": page_index
                            }
            if last_page_pointer is not None and first_page_pointer is not None:
                break

        adding = False
        new_time_ranges = []
        for t in self.time_ranges:
            new_time_range = TimeRange(t.index)
            for page_index, page in enumerate(t.pages):
                new_page = PageDetail({
                    "filename": page.filename,
                    "year": page.year,
                    "lines": []
                })
                for line in page.lines:
                    if t.index == first_page_pointer["tr_index"] and page_index == first_page_pointer["page_index"]:
                        adding = True
                    if t.index == last_page_pointer["tr_index"] and page_index == last_page_pointer["page_index"]:
                        adding = False
                    if adding:
                        new_page.lines.append(line)
                new_time_range.add_page(new_page)
            new_time_ranges.append(new_time_range)

        new_time_ranges = [t for t in new_time_ranges if t.line_count() > 0]
        return Data(new_time_ranges)
