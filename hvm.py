from openai import AsyncOpenAI
import re
import os
import sys
import asyncio
import time

def parse_files_and_chunks(content):
    """
    Parse the input content into files and their chunks.
    Returns dict of filename -> list of (chunk_number, chunk_content) tuples
    """
    files = {}
    current_file = None
    current_chunk = []
    current_chunk_num = None
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for new file
        if line.startswith('./'):
            if current_file and current_chunk_num is not None and current_chunk:
                files[current_file].append((current_chunk_num, '\n'.join(current_chunk)))
            
            current_file = line[2:]  # Remove './'
            files[current_file] = []
            current_chunk = []
            current_chunk_num = None
            
        # Check for new chunk
        elif re.match(r'^#\d+:$', line):
            if current_file is None:
                raise ValueError("Found chunk before file declaration")
                
            if current_chunk_num is not None and current_chunk:
                files[current_file].append((current_chunk_num, '\n'.join(current_chunk)))
            
            current_chunk_num = line.strip('#:')
            current_chunk = []
            
        # Add content to current chunk
        elif current_chunk_num is not None and current_file is not None:
            current_chunk.append(line)
            
        i += 1
    
    # Don't forget the last chunk of the last file
    if current_file and current_chunk_num is not None and current_chunk:
        files[current_file].append((current_chunk_num, '\n'.join(current_chunk)))
    
    return files

def create_deepseek_messages(filename, block_num, content, task):
    """Create the messages for the Deepseek API"""
    system_prompt = f"""
You are an AI assistant specialized in analyzing code blocks for potential refactoring needs. Your task is to examine each code block provided and determine if it requires refactoring to accomplish a specific task.

Here's the Type.hs of the codebase to be refactor:

<Type.hs>
module HVML.Type where

import Data.Map.Strict as MS
import Data.Word
import Foreign.Ptr

-- Core Types
-- ----------

data Core
  = Var String -- x
  | Ref String Word64 [Core] -- @fn
  | Era -- *
  | Lam String Core -- λx(F)
  | App Core Core -- (f x)
  | Sup Word64 Core Core -- &L{{a b}}
  | Dup Word64 String String Core Core -- ! &L{{a b}} = v body
  | Ctr Word64 [Core] -- #Ctr{{a b ...}}
  | Mat Core [(String,Core)] [(String,[String],Core)] -- ~ v {{ #A{{a b ...}}: ... #B{{a b ...}}: ... ... }}
  | U32 Word32 -- 123
  | Chr Char -- 'a'
  | Op2 Oper Core Core -- (+ a b)
  | Let Mode String Core Core -- ! x = v body
  deriving (Show, Eq)

data Mode
  = LAZY
  | STRI
  | PARA
  deriving (Show, Eq, Enum)

data Oper
  = OP_ADD | OP_SUB | OP_MUL | OP_DIV
  | OP_MOD | OP_EQ  | OP_NE  | OP_LT
  | OP_GT  | OP_LTE | OP_GTE | OP_AND
  | OP_OR  | OP_XOR | OP_LSH | OP_RSH
  deriving (Show, Eq, Enum)

-- A top-level function, including:
-- - copy: true when ref-copy mode is enabled
-- - args: a list of (isArgStrict, argName) pairs
-- - core: the function's body
-- Note: ref-copy improves C speed, but increases interaction count
type Func = ((Bool, [(Bool,String)]), Core)

data Book = Book
  {{ idToFunc :: MS.Map Word64 Func
  , idToName :: MS.Map Word64 String
  , idToLabs :: MS.Map Word64 (MS.Map Word64 ())
  , nameToId :: MS.Map String Word64
  , ctrToAri :: MS.Map String Int
  , ctrToCid :: MS.Map String Word64
  }} deriving (Show, Eq)
</Type.hs>

Here's the task that the code needs to accomplish:

<task>
{task}
</task>

For each code block you analyze, follow these steps:

1. Examine the code block and the file name it belongs to.
2. Analyze whether the code block needs refactoring to complete the task described above.
3. After your analysis, provide your final decision in <decision> tags.

Your final decision should be one of the following:
- If the block needs refactoring, respond only with the block number.
- If it doesn't need refactoring, respond with 'no'.

Here's an example of how your response should be structured:

<decision>
[Your decision goes here - either the block number or 'no']
</decision>

Remember:
- Each code block is part of a larger file.
- The refactoring needs to include updating comments and documentation, not only whether block needs refactoring to complete the given task.
- Do not suggest actual refactoring changes; only identify if refactoring is needed.
- Respond with the decision only.

Please proceed with your analysis when presented with a code block.
    """
    
    user_content = f"""Filename: {filename}
Block #{block_num}:
{content}"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

async def analyze_chunk(client, filename, chunk_num, content, task):
    """Analyze a single chunk using the Deepseek API"""
    messages = create_deepseek_messages(filename, chunk_num, content, task)
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1,
            max_tokens=50,
            stream=False
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract content from decision tags
        match = re.search(r'<decision>(.*?)</decision>', response_text, re.DOTALL)
        if match:
            decision = match.group(1).strip()
            if decision != 'no':
                return filename, chunk_num
        return None
        
    except Exception as e:
        print(f"Error processing {filename} chunk {chunk_num}: {str(e)}")
        return None

async def analyze_chunks(files, api_key, task):
    """
    Analyze chunks from all files using the Deepseek API concurrently.
    Returns dict of filename -> list of block numbers that need refactoring.
    """
    results = {filename: [] for filename in files}
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    # Create tasks for all chunks
    tasks = []
    for filename, chunks in files.items():
        for chunk_num, content in chunks:
            task_obj = analyze_chunk(client, filename, chunk_num, content, task)
            tasks.append(task_obj)
    
    # Run all tasks concurrently
    start_time = time.time()
    print(f"\nAnalyzing {len(tasks)} chunks...")
    
    chunk_results = await asyncio.gather(*tasks)
    
    # Process results
    for result in chunk_results:
        if result:
            filename, chunk_num = result
            results[filename].append(chunk_num)
    
    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds")
    
    return results

async def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <task>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    task = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Get API key from environment variable
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: Please set DEEPSEEK_API_KEY environment variable")
        sys.exit(1)
    
    # Read the input file
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        sys.exit(1)
    
    if not content:
        print("Error: Input file is empty")
        sys.exit(1)
    
    # Parse the chunks
    try:
        files = parse_files_and_chunks(content)
    except ValueError as e:
        print(f"Error parsing input: {str(e)}")
        sys.exit(1)
    
    # Analyze the chunks
    results = await analyze_chunks(files, api_key, task)
    
    # Print results
    print("\nAnalysis Results:")
    print("----------------")
    found_blocks = False
    output_blocks = []
    for filename, blocks in results.items():
        if blocks:
            found_blocks = True
            # print(f"\n{filename}:")
            for block_num in sorted(blocks, key=int):
                output_blocks.append(int(block_num))
                # print(f"  Block #{block_num}")
    
    if not found_blocks:
        print("\nNo blocks need refactoring in any file.")
    return sorted(output_blocks)

if __name__ == "__main__":
    asyncio.run(main())