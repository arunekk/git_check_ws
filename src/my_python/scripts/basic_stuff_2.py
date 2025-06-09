# def extract_values_below_threshold(file_path, threshold=0.5):
#     result = []

#     with open(file_path, 'r') as file:
#         for line in file:
#             # Remove leading/trailing whitespace and split by commas
#             values = line.strip().split(',')
#             for value in values:
#                 try:
#                     num = float(value.strip())
#                     if num < threshold:
#                         result.append(num)
#                 except ValueError:
#                     continue  # skip invalid values

#     return result

# # Example usage
# csv_file = 'camera_values.csv'  # Replace with your actual file path
# filtered = extract_values_below_threshold(csv_file)

# print("Values less than 0.5:", filtered)
# print("Number of values less than 0.5:", len(filtered))
def process_values(file_path, threshold=0.5):
    all_values = []
    filtered_values = []

    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            for value in values:
                try:
                    num = float(value.strip())
                    all_values.append(num)
                    if num < threshold:
                        filtered_values.append(num)
                except ValueError:
                    continue  # Skip non-numeric entries

    if not all_values:
        return {
            "all_values": [],
            "min": None,
            "max": None,
            "filtered_values": []
        }

    return {
        "all_values": all_values,
        "min": min(all_values),
        "max": max(all_values),
        "filtered_values": filtered_values
    }

# Example usage
# csv_file = 'laser_values_2.csv'  # Replace with your file path
csv_file = 'camera_values_2.csv'  # Replace with your file path
result = process_values(csv_file)

# Output
print("All values count:", len(result["all_values"]))
print("Minimum value:", result["min"])
print("Maximum value:", result["max"])
print("Values less than 0.5:", result["filtered_values"])
print("Number of values less than 0.5:", len(result["filtered_values"]))
