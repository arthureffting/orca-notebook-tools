from src.breakpoint import Breakpoint
from src.line_detail import LineDetail
import datetime


class PageDetail:

    def __init__(self, page_data):
        self.json_data = page_data
        self.filename = page_data["filename"]
        self.year = page_data["year"]
        self.lines = [LineDetail(self, line) for line in page_data["lines"]]
        self.range_index = "_".join(self.filename.split("_")[0:2])

    def copy_with_tape_mentions(self, tapes=None):
        return PageDetail({
            "filename": self.filename,
            "year": self.year,
            "lines": [{
                "line": line.index,
                "text": line.text,
            } for line in self.lines if line.has_tape_mention(tapes)]
        })

    def copy_with_keyword_mentions(self, keyword, prob_threshold=0.75):
        return PageDetail({
            "filename": self.filename,
            "year": self.year,
            "lines": [{
                "line": line.index,
                "text": line.text,
            } for line in self.lines if line.contains(keyword, prob_threshold)]
        })

    def copy_with_pod_mentions(self, pods=None):
        return PageDetail({
            "filename": self.filename,
            "year": self.year,
            "lines": [{
                "line": line.index,
                "text": line.text,
            } for line in self.lines if line.has_pod_mention(pods)]
        })

    def as_json(self):
        return {
            "lines": [line.as_json() for line in self.lines],
            "filename": self.filename,
            "year": self.year,
        }

    @staticmethod
    def from_json(page_json):
        page = PageDetail({
            "lines": [],
            "filename": page_json["filename"],
            "year": page_json["year"]
        })
        for line_json in page_json["lines"]:
            page.lines.append(LineDetail.from_json(page, line_json))
        return page
