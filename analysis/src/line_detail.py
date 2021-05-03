import datetime
import json
import re

from analysis.src.utils import contains


class LineDetail:

    def __init__(self, page, line_data, only_basic=False):
        self.page = page
        self.index = line_data["line"]
        self.text = line_data["text"]

        if not only_basic:
            self.timestamp = None
            self.date_regex = re.compile(
                "^([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|,|'|`|)([1-9]|0[1-9]|1[0-2])(\.|-|,|'|`|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])$|^([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|,|'|`|/)([1-9]|0[1-9]|1[0-2])(\.|-|,|'|`|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])$")
            self.family_regex = re.compile("([A-Z][0-9]([0-9])?)")
            self.date = None

            try:
                dates_found = self.date_regex.findall(self.text)
                date_result = dates_found[0]
                date_parts = []
                for date_part in date_result:
                    if str.isnumeric(date_part):
                        date_parts.append(date_part)
                day = int(date_parts[1])
                month = int(date_parts[0])
                year = int(date_parts[2])
                year = year if int(year) > 1900 else year + (2000 if year < 10 else 1900)
                self.date = datetime.datetime(year, month, day).date()
            except:
                self.date = None

            try:
                self.pod_mentions = [p[0] for p in self.family_regex.findall(self.text.replace(" ", ""))]
            except:
                self.pod_mentions = []

            self.tape = None
            with open("possible_tapes.json") as tape_data:
                tapes = json.load(tape_data)

                for tape in tapes:
                    lower_text = self.text.replace(" ", "")
                    if tape in lower_text and (tape + "m") not in lower_text:
                        start_result = contains(self.text, "start", probability=0.75)
                        zero_percent_result = "0%" in lower_text and "100%" not in lower_text and "10%" not in lower_text
                        end_result = "end" in lower_text
                        hundred_percent_result = "100%" in lower_text or "00%" in lower_text or "10%" in lower_text
                        self.tape = {
                            "number": tape[:-1],
                            "side": tape[-1],
                            "starts": start_result or zero_percent_result,
                            "ends": end_result or hundred_percent_result
                        }
                        break

    def has_date(self):
        return self.date is not None

    @staticmethod
    def from_json(page, line_json):
        line = LineDetail(page, {
            "line": line_json["index"],
            "text": line_json["text"]
        }, only_basic=True)
        line.pod_mentions = line_json["pod_mentions"] if "pod_mentions" in line_json else []
        line.tape = line_json["tape"] if "tape" in line_json else None
        line.date = datetime.datetime.strptime(line_json["date"], "%Y-%m-%d") if "date" in line_json else None
        return line

    def as_json(self):
        data = {
            "year": self.page.year,
            "filename": self.page.filename,
            "index": self.index,
            "text": self.text,
            "pod_mentions": self.pod_mentions,
            "tape": self.get_tape_mention()
        }
        if self.has_date():
            data["date"] = str(self.date)
        return data

    def get_date(self):
        return self.date

    def has_tape_mention(self, tapes=None):
        has_tape = self.tape is not None
        if not has_tape:
            return False
        if tapes is None:
            return True
        number_in_tapes = self.tape["number"] in tapes
        number_and_side_in_tapes = (self.tape["number"] + self.tape["side"]) in tapes
        return has_tape if tapes is None else has_tape and (number_in_tapes or number_and_side_in_tapes)

    def has_pod_mention(self, pods=None):
        has_pods = self.pod_mentions is not None and len(self.pod_mentions) > 0
        if not has_pods:
            return False

        if pods is None:
            return has_pods

        result = False
        if has_pods:
            for desired_pod in pods:
                specific = len(desired_pod) > 1 and str.isnumeric(desired_pod[1:])
                if specific:
                    if any([mentioned_pod == desired_pod for mentioned_pod in self.pod_mentions]):
                        result = True
                        break
                else:
                    only_families = set([mentioned_pod[0] for mentioned_pod in self.pod_mentions])
                    if any([family == desired_pod[0] for family in only_families]):
                        result = True
                        break
        return result

    def get_tape_mention(self):
        return self.tape

    def has_timestamp(self):
        return self.timestamp is not None

    def get_timestamp(self):
        return self.timestamp

    def contains(self, keyword, probability):
        return contains(self.text, keyword, probability)

    def equals(self, other):
        return self.page.filename == other.page.filename and self.text == other.text and self.index == other.index
