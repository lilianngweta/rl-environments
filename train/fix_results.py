#!/usr/bin/env python3
"""Fix results.jsonl by replacing null content with empty strings."""

import json
import sys

def fix_results(input_path, output_path):
    """Read results, fix null content, write to output."""
    fixed_count = 0
    total_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            total_count += 1
            data = json.loads(line)
            
            # Check completion content
            if 'completion' in data and isinstance(data['completion'], list):
                for msg in data['completion']:
                    if 'content' in msg and msg['content'] is None:
                        msg['content'] = ""
                        fixed_count += 1
            
            outfile.write(json.dumps(data) + '\n')
    
    print(f"Fixed {fixed_count} null content fields out of {total_count} samples")
    return fixed_count

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: fix_results.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    fix_results(input_path, output_path)

# Made with Bob
