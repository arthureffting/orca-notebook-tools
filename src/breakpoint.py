class Breakpoint:
    """
    A breakpoint is any line which contains a date.
    It can be valid or invalid, meaning it lies within the range established by the filename.
    Ex: aug1292_oct_1293.png can only contain valid breakpoints between 12.08.1992 and 12.10.1993.
    """

    def __init__(self, range, line):
        self.range = range
        self.line = line

    def get_date(self):
        return self.line.get_date()

    def is_valid(self):
        """
        Checks whether the date of the breakpoint is within the boundaries established by the file name.
        :return: True if date is valid, False otherwise.
        """
        return self.range.start < self.get_date() < self.range.end
