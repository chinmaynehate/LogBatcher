import os
import re
import argparse
from collections import Counter, defaultdict


def parse_log_levels(log_file):
    """
    Parse general log files to count occurrences of each log level and extract details.

    Args:
        log_file (str): Path to the log file.

    Returns:
        tuple: (log_counts, log_details)
            - log_counts: Counter with counts of each log level.
            - log_details: Dictionary with segregated details for each log level.
    """
    log_level_pattern = r'\[(debug|info|notice|warning|error|critical)\]'
    log_counts = Counter({level.upper(): 0 for level in ["debug", "info", "notice", "warning", "error", "critical"]})
    log_details = {level.upper(): [] for level in ["debug", "info", "notice", "warning", "error", "critical"]}

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(log_level_pattern, line, re.IGNORECASE)
                if match:
                    log_level = match.group(1).upper()
                    log_counts[log_level] += 1

                    # Extract the message part
                    message = line.split(match.group(0))[-1].strip()
                    log_details[log_level].append({"message": message})
                else:
                    print(f"Unmatched line: {line.strip()}")  # Debugging unmatched lines

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None

    return log_counts, log_details


def parse_blg_logs(log_file_path):
    log_summary = defaultdict(int)
    log_details = defaultdict(list)  # Use this to store detailed log messages

    log_patterns = {
        "kernel_info": re.compile(r"RAS KERNEL INFO"),
        "kernel_fatal": re.compile(r"RAS KERNEL FATAL"),
        "app_fatal": re.compile(r"RAS APP FATAL"),
        "double_hummer": re.compile(r"double-hummer alignment exceptions"),
        "instruction_cache": re.compile(r"instruction cache parity error corrected"),
    }

    with open(log_file_path, 'r') as file:
        for line in file:
            for log_type, pattern in log_patterns.items():
                if pattern.search(line):
                    log_summary[log_type] += 1
                    log_details[log_type].append({"message": line.strip()})  # Store as a dictionary

    return log_summary, log_details

def parse_hpc_logs(log_file):
    """
    Parse HPC logs to extract details based on custom format.

    Args:
        log_file (str): Path to the HPC log file.

    Returns:
        tuple: (log_counts, log_details)
            - log_counts: Counter with counts of each log category.
            - log_details: Dictionary with segregated details for each category.
    """
    log_counts = Counter()
    log_details = defaultdict(list)

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(maxsplit=5)  # Split into at most 6 parts
                if len(parts) < 6:
                    # Skip lines that do not match the expected format
                    print(f"Unmatched line: {line.strip()}")
                    continue

                # Extract relevant fields
                timestamp_id = parts[0]  # Example: `134681`
                node = parts[1]  # Example: `node-246`
                category = parts[2]  # Example: `unix.hw`
                event_type = parts[3]  # Example: `state_change.unavailable`
                timestamp = parts[4]  # Example: `1077804742`
                message = parts[5]  # Remaining part is the message

                log_key = f"{category}.{event_type}"  # Combine category and event type for unique key
                log_counts[log_key] += 1
                log_details[log_key].append({
                    "timestamp_id": timestamp_id,
                    "node": node,
                    "timestamp": timestamp,
                    "message": message,
                })

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None

    return log_counts, log_details


def parse_android_logs(log_file):
    """
    Parse the Android log file to count occurrences of each log level and extract details.

    Args:
        log_file (str): Path to the Android log file.

    Returns:
        tuple: (log_counts, log_details)
            - log_counts: Counter with counts of each log level.
            - log_details: Dictionary with segregated details for each log level.
    """
    full_pattern = r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+\d+\s+\d+\s+([VDIWEF])\s+[^\s]+:\s+(.*)'
    log_counts = Counter({level: 0 for level in ["VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"]})
    log_details = {level: [] for level in ["VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"]}

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.match(full_pattern, line)
                if match:
                    timestamp, level_char, message = match.groups()
                    log_level = {
                        "V": "VERBOSE",
                        "D": "DEBUG",
                        "I": "INFO",
                        "W": "WARNING",
                        "E": "ERROR",
                        "F": "FATAL",
                    }.get(level_char, "UNKNOWN")

                    log_counts[log_level] += 1
                    log_details[log_level].append({"timestamp": timestamp, "message": message})
                else:
                    print(f"Unmatched line: {line.strip()}")  # Debug unmatched lines

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None

    return log_counts, log_details

def parse_healthapp_logs(log_file):
    """
    Parse HealthApp logs to extract details based on custom format.

    Args:
        log_file (str): Path to the HealthApp log file.

    Returns:
        tuple: (log_counts, log_details)
            - log_counts: Counter with counts of each log type (e.g., 'Step_LSC', 'Step_SPUtils').
            - log_details: Dictionary with segregated details for each log type.
    """
    log_counts = Counter()
    log_details = defaultdict(list)

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                # Split the log line by '|'
                parts = line.strip().split('|')
                if len(parts) < 4:
                    # Skip lines that do not match the expected format
                    print(f"Unmatched line: {line.strip()}")
                    continue

                # Extract type and message
                log_type = parts[1]  # Example: 'Step_LSC', 'Step_SPUtils'
                message = "|".join(parts[3:])  # Join the rest as the message

                log_counts[log_type] += 1
                log_details[log_type].append({"message": message})

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None

    return log_counts, log_details


def parse_hadoop_logs(log_file):
    """
    Parse Hadoop logs to count occurrences of each log level and extract details.

    Args:
        log_file (str): Path to the Hadoop log file.

    Returns:
        tuple: (log_counts, log_details)
            - log_counts: Counter with counts of each log level.
            - log_details: Dictionary with segregated details for each log level.
    """
    log_patterns = {
        "ERROR": re.compile(r"ERROR"),
        "WARN": re.compile(r"WARN"),
        "INFO": re.compile(r"INFO"),
        "DEBUG": re.compile(r"DEBUG"),
    }
    log_counts = Counter({level: 0 for level in log_patterns.keys()})
    log_details = {level: [] for level in log_patterns.keys()}

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                for level, pattern in log_patterns.items():
                    if pattern.search(line):
                        log_counts[level] += 1
                        log_details[level].append({"message": line.strip()})

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None

    return log_counts, log_details

def parse_hdfs_logs(log_file):
    """
    Parse HDFS logs to count occurrences of each log level and extract details.

    Args:
        log_file (str): Path to the HDFS log file.

    Returns:
        tuple: (log_counts, log_details)
            - log_counts: Counter with counts of each log level.
            - log_details: Dictionary with segregated details for each log level.
    """
    log_patterns = {
        "ERROR": re.compile(r"ERROR"),
        "WARN": re.compile(r"WARN"),
        "INFO": re.compile(r"INFO"),
        "DEBUG": re.compile(r"DEBUG"),
    }
    log_counts = Counter({level: 0 for level in log_patterns.keys()})
    log_details = {level: [] for level in log_patterns.keys()}

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                for level, pattern in log_patterns.items():
                    if pattern.search(line):
                        log_counts[level] += 1
                        log_details[level].append({"message": line.strip()})

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None

    return log_counts, log_details





def display_summary_with_analysis(log_counts, log_details, total_logs, output_file=None):
    """
    Display a detailed summary with additional analyses for developers.

    Args:
        log_counts (Counter): Counts of log levels.
        log_details (dict): Segregated details for each log level.
        total_logs (int): Total number of logs.
        output_file (str, optional): File path to save the summary. Defaults to None.
    """
    summary = "Log Level Summary:\n"
    summary += "-" * 30 + "\n"

    # Display top 5 log levels by count
    for level, count in log_counts.most_common(5):
        percentage = (count / total_logs) * 100 if total_logs > 0 else 0
        summary += f"{level:<30}{count:>5} ({percentage:>5.1f}%)\n"

    summary += "-" * 30 + "\n"
    summary += f"Total Logs: {total_logs}\n"

    # Full Message Log (First 3 per Level)
    summary += "\nDetails for Top Log Types:\n"
    for level, logs in list(log_details.items())[:3]:  # Limit to 3 log types
        summary += f"\n{level}:\n"
        for idx, log in enumerate(logs[:3]):  # Limit to 3 messages per level
            summary += f"  {idx + 1:>2}. {log['message']}\n"
        if len(logs) > 3:
            summary += f"  ... ({len(logs) - 3} more logs)\n"

    print(summary)

    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as file:
                file.write(summary)
            print(f"Summary saved to {output_file}")
        except Exception as e:
            print(f"Error saving summary: {e}")


import re
from collections import Counter, defaultdict

def parse_linux_logs(log_file):
    """
    Parse Linux logs to extract structured information and counts.

    Args:
        log_file (str): Path to the Linux log file.

    Returns:
        tuple: (log_counts, log_details)
            - log_counts: Counter with counts of each log event.
            - log_details: Dictionary with segregated details for each log event type.
    """
    log_counts = Counter()
    log_details = defaultdict(list)
    
    log_pattern = re.compile(
        r"^(?P<date>\w+ \d+ \d{2}:\d{2}:\d{2}) (?P<hostname>\S+) (?P<process>\S+)\[(?P<pid>\d+)\]: (?P<event>.*)"
    )

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                match = log_pattern.match(line.strip())
                if match:
                    data = match.groupdict()
                    event_type = data["event"].split(";")[0].strip()  # Use the first part of the event message as type
                    log_counts[event_type] += 1
                    log_details[event_type].append({
                        "date": data["date"],
                        "hostname": data["hostname"],
                        "process": data["process"],
                        "pid": data["pid"],
                        "message": data["event"],
                    })
                else:
                    print(f"Unmatched line: {line.strip()}")

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None

    return log_counts, log_details

import re
from collections import Counter, defaultdict


def parse_mac_logs(log_file):
    """
    Parse Mac logs to extract structured information and counts.
    Handles various formats found in macOS logs.
    """
    log_counts = Counter()
    log_details = defaultdict(list)

    # General pattern to match the log format
    log_pattern = re.compile(
        r"^(?P<date>\w{3} +\d+ \d{2}:\d{2}:\d{2}) (?P<hostname>[\w.-]+) (?P<process>[\w\.\[\]]+)?(?:\[(?P<pid>\d+)\])?: (?P<message>.*)"
    )

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                match = log_pattern.match(line.strip())
                if match:
                    data = match.groupdict()
                    event_type = data["message"].split(":")[0].strip()  # Use the first part of the message for event type
                    log_counts[event_type] += 1
                    log_details[event_type].append({
                        "date": data["date"],
                        "hostname": data["hostname"],
                        "process": data.get("process", "Unknown"),
                        "pid": data.get("pid", "N/A"),
                        "message": data["message"],
                    })
                else:
                    # Log unmatched lines for debugging
                    print(f"Unmatched line: {line.strip()}")

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None

    return log_counts, log_details


import re
from collections import Counter, defaultdict

def parse_ssh_logs(log_file):
    """
    Parse SSH logs to extract invalid login attempts, break-in attempts, and other relevant events.

    Args:
        log_file (str): Path to the SSH log file.

    Returns:
        tuple: (log_counts, log_details)
            - log_counts: Counter with counts of each log event type.
            - log_details: Dictionary with segregated details for each log type.
    """
    log_counts = Counter()
    log_details = defaultdict(list)

    # Regex to parse log lines
    log_pattern = re.compile(
        r"^(?P<date>\w+ \d+ \d{2}:\d{2}:\d{2}) (?P<hostname>[\w.-]+) (?P<process>\S+): (?P<message>.*)"
    )

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                match = log_pattern.match(line.strip())
                if match:
                    data = match.groupdict()
                    message = data["message"]
                    
                    # Categorize the message
                    if "Invalid user" in message:
                        event_type = "Invalid user"
                    elif "authentication failure" in message:
                        event_type = "Authentication failure"
                    elif "Failed password" in message:
                        event_type = "Failed password"
                    elif "Connection closed" in message:
                        event_type = "Connection closed"
                    elif "POSSIBLE BREAK-IN ATTEMPT" in message:
                        event_type = "Break-in attempt"
                    else:
                        event_type = "Other"

                    # Increment count and store details
                    log_counts[event_type] += 1
                    log_details[event_type].append({
                        "date": data["date"],
                        "hostname": data["hostname"],
                        "process": data["process"],
                        "message": message,
                    })
                else:
                    print(f"Unmatched line: {line.strip()}")  # Debug unmatched lines

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None

    return log_counts, log_details


def determine_log_path(config, dataset):
    """
    Determine the log file path based on config and dataset.

    Args:
        config (str): Configuration name.
        dataset (str): Dataset name.

    Returns:
        str: Path to the log file.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_root, "datasets/loghub-2k")

    if dataset.lower() == "android":
        log_file = os.path.join(base_dir, "android", "Android_2k.log")
    elif dataset.lower() == "apache":
        log_file = os.path.join(base_dir, "apache", "apache_2k.log")
    elif dataset.lower() == "blg":
        log_file = os.path.join(base_dir, "BGL", "BGL_2k.log")
    elif dataset.lower() == "hadoop":
        log_file = os.path.join(base_dir, "hadoop", "hadoop_2k.log")
    elif dataset.lower() == "hdfs":
        log_file = os.path.join(base_dir, "HDFS", "HDFS_2k.log")
    elif dataset.lower() == "healthapp":
        log_file = os.path.join(base_dir, "HealthApp", "HealthApp_2k.log")
    elif dataset.lower() == "hpc":
        log_file = os.path.join(base_dir, "HPC", "HPC_2k.log")
    elif dataset.lower() == "linux":
        log_file = os.path.join(base_dir, "Linux", "Linux_2k.log")
    elif dataset.lower() == "mac":
        log_file = os.path.join(base_dir, "Mac", "Mac_2k.log")
    elif dataset.lower() == "ssh":
        log_file = os.path.join(base_dir, "OpenSSH", "OpenSSH_2k.log")    

    else:
        log_file = os.path.join(base_dir, dataset, f"{dataset}_2k.log")

    abs_path = os.path.abspath(log_file)
    print(f"Looking for log file at: {abs_path}")
    if not os.path.exists(log_file):
        print(f"Log file does not exist at: {abs_path}")
        return None
    return log_file


def main():
    """
    Main function to process log files, generate summaries, and visualize log levels.
    """
    parser = argparse.ArgumentParser(description="Generate a log level summary.")
    parser.add_argument("--config", type=str, required=True, help="Configuration name (e.g., 'test').")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'Apache', 'Android', 'BLG', 'Hadoop', 'HealthApp').")
    parser.add_argument("--output-summary", type=str, default="outputs/summary.txt", help="Path to save the log level summary.")

    args = parser.parse_args()

    # Determine log file path
    log_file = determine_log_path(args.config, args.dataset)
    if not log_file:
        return

    # Parse logs based on dataset
    if args.dataset.lower() == "android":
        log_counts, log_details = parse_android_logs(log_file)
    elif args.dataset.lower() == "apache":
        log_counts, log_details = parse_log_levels(log_file)
    elif args.dataset.lower() == "blg":
        log_counts, log_details = parse_blg_logs(log_file)
    elif args.dataset.lower() == "hadoop":
        log_counts, log_details = parse_hadoop_logs(log_file)
    elif args.dataset.lower() == "hdfs":
        log_counts, log_details = parse_hdfs_logs(log_file)
    elif args.dataset.lower() == "healthapp":
        log_counts, log_details = parse_healthapp_logs(log_file)
    elif args.dataset.lower() == "hpc":
        log_counts, log_details = parse_hpc_logs(log_file)
    elif args.dataset.lower() == "linux":
        log_counts, log_details = parse_linux_logs(log_file)
    elif args.dataset.lower() == "mac":
        log_counts, log_details = parse_mac_logs(log_file)
    elif args.dataset.lower() == "ssh":
        log_counts, log_details = parse_mac_logs(log_file)
    else:
        print(f"Dataset {args.dataset} not supported.")
        return

    if not log_counts or not log_details:
        print("No log levels found in the log file or file not readable.")
        return

    # Total logs
    total_logs = sum(log_counts.values())

    # Display or save the summary
    display_summary_with_analysis(log_counts, log_details, total_logs, output_file=args.output_summary)


if __name__ == "__main__":
    main()
