import os
import re
import yaml
import argparse
from collections import Counter, defaultdict


class LogIssueTagger:
    """
    A class to tag and summarize issues and anomalies in log files.
    """

    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.tagging_threshold = self.config.get("tagging_threshold", 10)  # Adjusted threshold
        self.patterns = self.load_patterns()

    @staticmethod
    def load_config(config_path):
        """Load configuration from a YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error reading configuration file: {e}")
            return {}

    def load_patterns(self):
        """Define regex patterns for potential issues."""
        return {
            "default": [
                r".*(ERROR|WARN|FATAL|CRITICAL|FAIL|EXCEPTION).*",
                r".*(Exception in thread).*",
                r".*(Cannot|Failed to|Unable to|Timed out|Refused|No such file|Not found|Permission denied|Segmentation fault).*",
                r".*(panic|abort|trap|illegal instruction|core dumped).*",
                r".*(Traceback).*",
                r".*(Caused by:).*",
                r".*(Invalid|Unexpected|Unknown|Corrupt|Bad|Mismatch|Conflict).*",
            ],
        }

    def parse_log(self, log_file, dataset):
        """Parse log files to track potential issues."""
        patterns = self.patterns.get(dataset.lower(), self.patterns["default"])
        counts = Counter()
        details = defaultdict(list)

        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    line = line.strip()
                    message = self.standardize_message(line)
                    if message:
                        counts[message] += 1
                        details[message].append(line)
        except FileNotFoundError:
            print(f"Log file not found: {log_file}")
            return {}
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
            return {}

        # Filter high-priority issues
        tagged_issues = {
            message: {
                "count": count,
                "priority": "high" if count >= self.tagging_threshold else "normal",
                "examples": details[message][:3],
            }
            for message, count in counts.items()
        }

        return tagged_issues

    def standardize_message(self, message):
        """
        Standardize messages and exclude known normal lines.
        """
        # Exclude lines that are known to be normal or informational
        normal_patterns = [
            r".*(INFO|DEBUG|TRACE).*",
            r".*Starting up.*",
            r".*Shutting down.*",
            r".*Connection established.*",
            r".*Heartbeat.*",
            r".*Polling.*",
        ]
        for pattern in normal_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return None  # Exclude normal lines

        # Standardize the message
        message = re.sub(r"\b0x[0-9a-fA-F]+\b", "<HEX>", message)
        message = re.sub(r"\b\d+\b", "<NUM>", message)
        message = re.sub(r"(/[^/ ]*)+/?", "<PATH>", message)
        message = re.sub(r"\b[\w.-]+?@\w+?\.\w+?\b", "<EMAIL>", message)
        message = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", message)
        return message.lower().strip()

    def display_tagged_issues(self, tagged_issues, output_file=None):
        """Display and optionally save tagged issues."""
        high_priority = {k: v for k, v in tagged_issues.items() if v["priority"] == "high"}
        if not high_priority:
            print("No high-priority issues found.")
            return

        summary = "High-Priority Issues Summary:\n"
        summary += "-" * 60 + "\n"

        for issue, details in sorted(high_priority.items(), key=lambda x: -x[1]["count"]):
            summary += f"Issue: {issue}\n"
            summary += f"  Count: {details['count']}\n"
            summary += f"  Priority: {details['priority']}\n"
            summary += f"  Examples:\n"
            for example in details["examples"]:
                summary += f"    - {example}\n"
            summary += "-" * 60 + "\n"

        print(summary)

        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(summary)
                print(f"Tagged issues saved to {output_file}")
            except Exception as e:
                print(f"Error saving tagged issues: {e}")


def process_all_datasets(base_dir, config_path, output_dir):
    """Process all datasets in the specified base directory."""
    tagger = LogIssueTagger(config_path)

    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if os.path.isdir(dataset_path):
            log_file = None
            for file_name in os.listdir(dataset_path):
                if file_name.endswith("_2k.log") or file_name.endswith(".log"):
                    log_file = os.path.join(dataset_path, file_name)
                    break
            if log_file and os.path.exists(log_file):
                print(f"Processing {dataset}...")
                tagged_issues = tagger.parse_log(log_file, dataset)
                if tagged_issues:
                    output_file = os.path.join(output_dir, f"{dataset}_tagged_issues.txt")
                    tagger.display_tagged_issues(tagged_issues, output_file=output_file)
                else:
                    print(f"No high-priority issues found in {dataset} logs.")
            else:
                print(f"No log file found for {dataset} in {dataset_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all datasets in the Loghub-2k directory.")
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory containing the datasets.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save the summaries.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    process_all_datasets(args.base_dir, args.config, args.output_dir)
