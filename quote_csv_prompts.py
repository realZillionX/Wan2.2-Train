import argparse
import csv
import sys

def quote_prompts(input_path, output_path):
    print(f"Reading from {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as fin:
            # Detect format or just assume standard CSV
            reader = csv.DictReader(fin)
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                print("Error: Empty CSV or no headers found.")
                return
                
            print(f"Columns found: {fieldnames}")
            
            if 'prompt' not in fieldnames:
                print("Warning: 'prompt' column not found. All columns will still be quoted.")

            with open(output_path, 'w', encoding='utf-8', newline='') as fout:
                # Use QUOTE_ALL to force quotes around EVERYTHING, ensuring prompts are quoted.
                writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                writer.writeheader()
                
                count = 0
                for row in reader:
                    # Clean up prompt if needed (e.g. strip newlines if they are messy, but CSV handles internal newlines fine)
                    if 'prompt' in row and row['prompt']:
                        row['prompt'] = row['prompt'].strip()
                    writer.writerow(row)
                    count += 1
                    
        print(f"Success! Processed {count} rows. All fields including 'prompt' are now wrapped in \"\".")
        print(f"Saved to: {output_path}")

    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Force quote all fields in a CSV file.")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("output", help="Output CSV file path")
    args = parser.parse_args()
    
    quote_prompts(args.input, args.output)
