# summarize.py
A recursive text summarizing script.

## IPO

### Input
The script expects the filename of a UTF-8 encoded text file as a command line argument.

### Processing
The script will summarise the contents of the input text file using a LLM.

The langchain RecursiveCharacterTextSplitter is used to split the input into chunks with a maximum size of CHUNK_SIZE tokens and an overlap of CHUNK_OVERLAP tokens.

The langchain.llms module is used to interface with the LLM.

In the example, the Oobabooga text-generation webui API is used with the langchain.llms TextGen class.

To ensure an accurate measurement of the chunk size during splitting, the RecursiveCharacterTextSplitter is instrumented with a length function based on the respective tokenizer of the LLM.

If the input file is larger than the context of the LLM used for summarisation, the resulting partial summaries are also concatenated and summarised.

A fine-tuned summarisation prompt is used in the default configuration.

### Output
Finally, the script writes the summarisation result to a file.

### Example call:
```bash
$ time ./summarize -f file.txt
ic| input_filename: 'file.txt'
ic| output_filename: 'file-analysis.json'
...
Writing to file file-analysis.json.
./summarize.py -f file.txt  124.67s user 1.96s system 13% cpu 16:01.85 total
```

### Example output file:
```json
{
  "summary": "The narrative follows several individuals navigating life after an apocalypse has ravaged America. They form bonds, experience love and loss, struggle to endure under severe circumstances, and ultimately strive towards rebuilding society. The plot includes elements like friendship, romance, hardship, and perseverance."
}
```

## Limitations
* the tokenization of the entire input takes a substantial amount of time, even when using the transformers fast tokenizer
* with the current example implementation, the llm queries have to happen in series and cannot occur concurrently, because of limitations in the LLM worker (Oobabooga)
* with the current example implementation, the tokenization and the LLM query cannot happen concurrently, because of limitations in the script

### Conclusion
By using batched LLM inference instead and implementing asynchronous processing in the script, the overall performance could be improved.

## Known bugs
* error handling is missing
* logging is missing

## Pylint
```
$ pylint summarize.py
************* Module summarize
summarize.py:162:10: E1102: TextGen is not callable (not-callable)

------------------------------------------------------------------
Your code has been rated at 9.24/10 (previous run: 8.85/10, +0.39)
```
