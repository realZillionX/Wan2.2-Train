import argparse
import pandas as pd
import csv

def quote_prompts(input_path, output_path):
    print(f"Reading from {input_path}...")
    try:
        # Try reading with pandas
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Pandas read failed ({e}), trying standard csv module with flexible parsing...")
        # Fallback manual read if pandas fails on bad quotes
        # This is a robust fallback
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        df = pd.DataFrame(data)

    if 'prompt' not in df.columns:
        print("Error: Column 'prompt' not found!")
        return

    print(f"Found {len(df)} rows. Quoting 'prompt' column...")
    
    # Ensure all prompts are strings and stripped
    df['prompt'] = df['prompt'].astype(str).str.strip()
    
    # Save with quoting enabled for non-numeric (default) or ALL
    # DiffSynth-Studio uses pandas read_csv usually, which handles quoting fine.
    # The user specifically requested wrapping in "", so we force QUOTE_NONNUMERIC or QUOTE_ALL
    # We will use csv.QUOTE_ALL for the prompt column effectively by just saving.
    # But pandas to_csv does quoting automatically if needed.
    # If user wants *explicit* quotes visible in text editor even if simple string:
    # We can force quote_level.
    
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved to {output_path} with quoted strings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")
    args = parser.parse_args()
    
    quote_prompts(args.input, args.output)
