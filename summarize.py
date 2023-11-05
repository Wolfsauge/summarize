#!/usr/bin/env python3

import sys

import argparse
import json
import re

from dataclasses import dataclass
from icecream import ic  # type: ignore

from transformers import AutoTokenizer  # type: ignore

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import TextGen
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class CommandlineArguments:
    file: str


# @dataclass
# class WrappedTextGen(TextGen):
#     client: Optional[Any] = None


# OOBA_NON_STREAMING_API_URL = "http://somehost:5000"
OOBA_NON_STREAMING_API_URL = "http://localhost:5000"

# jondurbin/airoboros-m-7b-3.1.2
# tokenizer = AutoTokenizer.from_pretrained(
#     "jondurbin/airoboros-m-7b-3.1.2", use_fast=True
# )

# # tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

# PROMPT_TEMPLATE = """BEGININPUT
# {text_to_summarize}
# ENDINPUT
# BEGININSTRUCTION
# Summarize the input in about 130 words.
# ENDINSTRUCTION
# """

# jondurbin/airoboros-l2-70b-3.1.2
tokenizer = AutoTokenizer.from_pretrained(
    "jondurbin/airoboros-l2-70b-3.1.2", use_fast=True
)

# tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

PROMPT_TEMPLATE = """BEGININPUT
{text_to_summarize}
ENDINPUT
BEGININSTRUCTION
Summarize the input in about 130 words.
ENDINSTRUCTION
"""

# ehartford/dolphin-2.2.1-mistral-7b
# PROMPT_TEMPLATE = """<|im_start|>system
# You are Dolphin, a helpful AI assistant.<|im_end|>
# <|im_start|>user
# Summarize the input in about 130 words.
#
# Input:
# {text_to_summarize}<|im_end|>
# <|im_start|>assistant
# """

# tokenizer = AutoTokenizer.from_pretrained(
#     "ehartford/dolphin-2.2.1-mistral-7b", use_fast=True
# )

# size of chunks in units of tokens

# jondurbin/airoboros-m-7b-3.1.2
# CHUNK_SIZE = 4000
# jondurbin/airoboros-l2-70b-3.1.2
CHUNK_SIZE = 8000

# overlap of chunks in units of tokens

# jondurbin/airoboros-m-7b-3.1.2
# CHUNK_OVERLAP = 512
# jondurbin/airoboros-l2-70b-3.1.2
CHUNK_OVERLAP = 1024


def get_file_contents(my_filename: str) -> str:
    with open(my_filename, "r", encoding="utf-8") as my_fp:
        return my_fp.read()


def get_length_of_chunk_in_chars(chunk: str) -> int:
    return len(chunk)


def get_length_of_chunk_in_tokens(chunk: str) -> int:
    return len(tokenizer(chunk).input_ids)


# size of chunks in units of tokens
CHUNK_SIZE = 4000

# overlap of chunks in units of tokens
CHUNK_OVERLAP = 512

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "."],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=get_length_of_chunk_in_tokens,
    is_separator_regex=False,
)


def summarize_this(my_chunk: str, my_chain: LLMChain, depth: int) -> str:
    depth += 1

    # length_of_chunk_in_chars = get_length_of_chunk_in_chars(my_chunk)
    length_of_chunk_in_tokens = get_length_of_chunk_in_tokens(my_chunk)

    if length_of_chunk_in_tokens >= CHUNK_SIZE:
        # we need to split
        chunks = text_splitter.split_text(my_chunk)
        ic(len(chunks))
        partial_results = []
        for partial_chunk in chunks:
            partial_results.append(summarize_this(partial_chunk, my_chain, depth))
        my_result_string = "\n".join(partial_results)

        intermediate_result = summarize_this(my_result_string, my_chain, depth)
    else:
        # we can summarize
        intermediate_result = my_chain.run(my_chunk)
        ic(len(intermediate_result))
        ic(intermediate_result)

    my_result = intermediate_result.strip()

    return my_result


def main(my_args: CommandlineArguments) -> None:
    # determine output filename
    input_filename = my_args.file
    output_filename = re.sub("\\.txt$", "-analysis.json", my_args.file)
    ic(input_filename)
    ic(output_filename)

    # read input text file
    sample_text = get_file_contents(my_args.file)

    # providing llm_chain
    my_prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["text_to_summarize"]
    )
    llm = TextGen(model_url=OOBA_NON_STREAMING_API_URL, preset="Divine Intellect")
    llm_chain = LLMChain(prompt=my_prompt, llm=llm)

    # enter recursion
    my_result = {}
    depth = 0
    my_result["summary"] = summarize_this(sample_text, llm_chain, depth)

    # peek at result
    ic(my_result)

    # write output file
    print(f"Writing to file {output_filename}.")
    with open(output_filename, "w", encoding="utf-8") as my_fp:
        json.dump(my_result, my_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="myfile.txt",
        help="inspect file (default myfile.txt)",
    )
    parsed_args = CommandlineArguments(**vars(parser.parse_args()))
    main(parsed_args)
    sys.exit()
