import os
import random
import string

def generate_random_filename(extension=".txt"):
    """Generate a random filename with the given extension."""
    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"{random_name}{extension}"

def split_file_by_empty_lines(input_file, output_dir):
    """Split the input file into separate files based on empty lines."""
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Split content into chunks by empty lines
    chunks = []
    current_chunk = []

    for line in lines:
        if line.strip():  # Non-empty line
            current_chunk.append(line)
        elif current_chunk:  # Empty line after a chunk
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:  # Add the last chunk if it exists
        chunks.append(current_chunk)

    # Write each chunk to a separate file in the output directory
    for chunk in chunks:
        filename = generate_random_filename()
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.writelines(chunk)
        print(f"Created file: {output_path}")

def process_all_text_files(input_dir, output_dir, move_dir):
    """Process all .txt files in the input directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each .txt file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_dir, file_name)
            print(f"Processing file: {input_file_path}")
            split_file_by_empty_lines(input_file_path, output_dir)
            os.rename(input_directory + file_name, move_dir + file_name)

# Example usage
input_directory = "./data/"  # Replace with your input directory path
output_directory = "./data/"  # Replace with your output directory path
move_directory = "./data/originals/"
process_all_text_files(input_directory, output_directory, move_directory)
