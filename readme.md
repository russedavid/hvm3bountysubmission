# HVM3 Bounty Submission

This tool analyzes HVM3 code blocks to identify portions that need refactoring based on a specified task. It uses the Deepseek API to analyze code blocks and determines which blocks require modification.

## Prerequisites

- Python 3.7 or higher
- `openai` Python package
- Deepseek API key

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install openai
```
3. Set up your Deepseek API key as an environment variable:
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

## Input File Format

The input file should follow this format:

```
./filename1
#1:
[code block content]

#2:
[code block content]

./filename2
#1:
[code block content]
```

Each file section starts with `./` followed by the filename. Code blocks are numbered and start with `#n:` where n is the block number.

## Usage

Run the script with two arguments:
1. The path to your input file
2. The refactoring task in quotes

```bash
python hvm.py input_file.txt "replace 'λx body' syntax with '\\x body'"
```

### Example

```bash
python hvm.py code_blocks.txt "update all function comments to include parameter types"
```

## Output

The script will return a list of codeblock numbers needing refactoring
