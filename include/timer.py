import time


class Timer:
    logs = {}
    summary = {}
    counts = {}

    def __init__(self) -> None:
        pass

    def begin(self, label):
        assert isinstance(label, str), "The subsection label must be a string"
        if label in self.logs and "start" in self.logs[label]:
            raise RuntimeError(
                f"Timer for '{label}' has already been started and not ended"
            )
        self.logs[label] = {"start": time.time()}

    def end(self, label):
        assert isinstance(label, str), "The subsection label must be a string"
        if label not in self.logs or "start" not in self.logs[label]:
            raise ValueError(f"No start time found for label: {label}")
        self.logs[label]["end"] = time.time()
        elapsed_time = self.logs[label]["end"] - self.logs[label]["start"]

        # Add to summary and counts
        if label in self.summary:
            self.summary[label] += elapsed_time
            self.counts[label] += 1
        else:
            self.summary[label] = elapsed_time
            self.counts[label] = 1

        # Clean up log entry
        del self.logs[label]

    def print_summary(self, sort_by="time", descending=True):
        """
        Print a formatted table summary of all timed sections.
        sort_by: 'time', 'label', or 'count'
        """

        print("\n" + "-" * 80)
        print("Timer Summary")
        print("-" * 80)

        # Adjust column widths
        label_width = 30
        time_width = 18
        calls_width = 10
        avg_width = 18
        header = (
            f"{'Label':<{label_width}}"
            f"{'Total Time (s)':>{time_width}}"
            f"{'Calls':>{calls_width}}"
            f"{'Avg Time (s)':>{avg_width}}"
        )
        print(header)
        print("-" * 80)

        # Prepare data
        summary_data = [
            (
                label,
                self.summary[label],
                self.counts[label],
                self.summary[label] / self.counts[label],
            )
            for label in self.summary
        ]

        # Sort
        if sort_by == "label":
            summary_data.sort(key=lambda x: x[0], reverse=descending)
        elif sort_by == "count":
            summary_data.sort(key=lambda x: x[2], reverse=descending)
        else:  # default: sort_by == "time"
            summary_data.sort(key=lambda x: x[1], reverse=descending)

        # Print each row
        for label, total, count, avg in summary_data:
            row = (
                f"{label:<{label_width}}"
                f"{total:>{time_width}.4f}"
                f"{count:>{calls_width}}"
                f"{avg:>{avg_width}.4f}"
            )
            print(row)

        print("-" * 80)
