import re
import argparse
import os
from pathlib import Path

def convert_faq_to_markdown(input_path, output_path):
    """Reads a text file and attempts to convert plausible Q&A blocks to Markdown."""
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            content = infile.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return False
    except Exception as e:
        print(f"Error reading input file {input_path}: {e}")
        return False

    output_md = ""
    current_answer_blocks = []
    found_qa = False

    # Split into blocks based on one or more blank lines
    # This treats consecutive blank lines as a single separator
    blocks = re.split(r'\n\s*\n', content.strip()) 

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Check if the last line ends with a question mark
        lines = block.splitlines()
        last_line = lines[-1].strip() if lines else ""

        if last_line.endswith('?'):
            found_qa = True # Mark that we found at least one potential Q&A
            # --- Found a potential question --- 
            # 1. Write out the accumulated answer from previous blocks
            if current_answer_blocks:
                full_answer = "\n\n".join(current_answer_blocks)
                output_md += full_answer.strip() + "\n\n"
                current_answer_blocks = [] # Reset answer buffer
            
            # 2. Format the current block as the question heading
            # Simplification: Treat the whole block as the question for the heading
            # Alternatively, could just use last_line
            question_text = " ".join(line.strip() for line in lines) # Re-join lines just in case
            output_md += f"### {question_text}\n\n"
        
        else:
            # --- Not a question block --- 
            # Assume it's part of the answer to the previous question
            current_answer_blocks.append(block)

    # Write any remaining answer blocks at the end of the file
    if current_answer_blocks:
        full_answer = "\n\n".join(current_answer_blocks)
        output_md += full_answer.strip() + "\n\n"
        
    # If no potential Q&A was found using the '?' heuristic, maybe skip writing?
    if not found_qa:
        print(f"Warning: No blocks ending in '?' found in {input_path}. Skipping conversion, check format.")
        return True # Considered 'success' as no error, but nothing done.

    # --- Write the combined Markdown output --- 
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(output_md.strip())
        return True

    except Exception as e:
        print(f"Error writing output file {output_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert plain text FAQ files (heuristic Q? format) in a directory to Markdown.')
    parser.add_argument('input_dir', help='Path to the input directory containing .txt files.')
    parser.add_argument('output_dir', help='Path to the output directory for Markdown files.')

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory does not exist or is not a directory: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for .txt files in: {input_path}")
    txt_files = list(input_path.rglob('*.txt'))

    if not txt_files:
        print("No .txt files found.")
        return

    print(f"Found {len(txt_files)} .txt files. Starting heuristic conversion...")
    success_count = 0
    failure_count = 0
    skipped_count = 0 # Count files where no '?' was found

    for txt_file in txt_files:
        relative_path = txt_file.relative_to(input_path)
        md_file = output_path / relative_path.with_suffix('.md')
        
        print(f"Processing: {txt_file} -> {md_file}")
        md_file.parent.mkdir(parents=True, exist_ok=True)
        
        result = convert_faq_to_markdown(txt_file, md_file)
        if result:
            # Check if the output file is empty or only whitespace, which happens if no Q? was found
            try:
                if not Path(md_file).exists() or Path(md_file).stat().st_size == 0:
                    print(f"  -> Skipped (No Q? found)")
                    skipped_count += 1
                    # Optionally delete the empty file
                    if Path(md_file).exists(): 
                        try: 
                            os.remove(md_file)
                        except OSError:
                            pass # Ignore if deletion fails
                else:
                    success_count += 1
            except FileNotFoundError:
                 # If file wasn't created at all (e.g. empty input?) count as skipped
                 print(f"  -> Skipped (No output generated)")
                 skipped_count += 1
        else:
            failure_count += 1
            
    print("\nBatch conversion completed.")
    print(f"Successfully converted (with Q?): {success_count}")
    print(f"Skipped (No Q? found / empty): {skipped_count}")
    print(f"Failed (Error during processing): {failure_count}")

if __name__ == "__main__":
    main() 