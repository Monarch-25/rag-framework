"""
This file contains various document chunking strategies for the RAG pipeline.
"""
import re
import numpy as np
from typing import List, Optional, Any, Callable, Iterable, Type, TypeVar, Literal, AbstractSet, Collection, Sequence
from abc import ABC, abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from enum import Enum
import tiktoken
from attr import dataclass
import backoff
import anthropic
from tqdm import tqdm


# --- Base Class ---
class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Splits the input text into a list of strings.
        This method must be implemented by subclasses.
        """
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller documents.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        all_chunks = []
        for i, text in enumerate(texts):
            chunks = self.split_text(text)
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk, metadata=metadatas[i]))
        return all_chunks

# --- TextSplitter (from fixed_token_chunker) ---
logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")
class TextSplitter(BaseChunker, ABC):
    """Interface for splitting text into chunks."""

    def __init__( 
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator in the chunks
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                              every document
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_tiktoken_encoder(
        cls: Type[TS], 
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> TS:
        """Text splitter that uses tiktoken encoder to count length."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate max_tokens_for_prompt. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        def _tiktoken_encoder(text: str) -> int:
            return len(
                enc.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        if issubclass(cls, FixedTokenChunker):
            extra_kwargs = {
                "encoding_name": encoding_name,
                "model_name": model_name,
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_tiktoken_encoder, **kwargs)

# --- Recursive Token Chunker ---
def _split_text_with_regex(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]

class RecursiveTokenChunker(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)

# --- Fixed Token Chunker ---
@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: Callable[[List[int]], str]
    """ Function to decode a list of token ids to a string"""
    encode: Callable[[str], List[int]]
    """ Function to encode a string to a list of token ids"""


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits

class FixedTokenChunker(TextSplitter):
    """Splitting text to tokens using model tokenizer."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for FixedTokenChunker. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

# --- Kamradt Modified Semantic Chunker ---
class KamradtModifiedChunker(BaseChunker):
    """
    A chunker that splits text into chunks of approximately a specified average size
    based on semantic similarity. Adapted from Greg Kamradt's work.
    """
    def __init__(self, avg_chunk_size:int=400, min_chunk_size:int=50, embedding_function=None, length_function=len):
        if embedding_function is None:
            raise ValueError("KamradtModifiedChunker requires an embedding_function.")
            
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=length_function
        )
        self.avg_chunk_size = avg_chunk_size
        self.embedding_function = embedding_function
        self.length_function = length_function

    def combine_sentences(self, sentences, buffer_size=1):
        for i in range(len(sentences)):
            combined_sentence = ''
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '
            combined_sentence += sentences[i]['sentence']
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']
            sentences[i]['combined_sentence'] = combined_sentence
        return sentences

    def calculate_cosine_distances(self, sentences):
        # Simplified batching for this context
        combined_sentences = [s['sentence'] for s in sentences]
        embeddings = self.embedding_function(combined_sentences)
        
        embedding_matrix = np.array(embeddings)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = embedding_matrix / norms
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
        
        distances = []
        for i in range(len(sentences) - 1):
            distance = 1 - similarity_matrix[i, i + 1]
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance
        return distances, sentences

    def split_text(self, text: str) -> List[str]:
        sentences_strips = self.splitter.split_text(text)
        sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(sentences_strips)]
        sentences = self.combine_sentences(sentences, 3)
        
        distances, sentences = self.calculate_cosine_distances(sentences)
        
        total_tokens = sum(self.length_function(s['sentence']) for s in sentences)
        if total_tokens == 0: return [] # Handle empty text
        
        number_of_cuts = total_tokens // self.avg_chunk_size
        
        if not distances: return [text] # Handle case with 1 or 0 sentences

        # Binary search for the best threshold
        lower_limit, upper_limit = 0.0, 1.0
        distances_np = np.array(distances)
        
        for _ in range(100): # 100 iterations for binary search
            threshold = (upper_limit + lower_limit) / 2.0
            num_points_above_threshold = np.sum(distances_np > threshold)
            if num_points_above_threshold > number_of_cuts:
                lower_limit = threshold
            else:
                upper_limit = threshold
        
        final_threshold = (upper_limit + lower_limit) / 2.0
        indices_above_thresh = [i for i, x in enumerate(distances) if x > final_threshold] 
        
        start_index = 0
        chunks = []
        for index in indices_above_thresh:
            group = sentences[start_index:index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)
            
        return chunks

# --- Cluster Semantic Chunker ---
class ClusterSemanticChunker(BaseChunker):
    def __init__(self, embedding_function=None, max_chunk_size=400, min_chunk_size=50, length_function=len):
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=length_function,
            separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
            )
        
        if embedding_function is None:
            raise ValueError("ClusterSemanticChunker requires an embedding_function.")

        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size//min_chunk_size
        self.embedding_function = embedding_function
        
    def _get_similarity_matrix(self, embedding_function, sentences):
        BATCH_SIZE = 500
        N = len(sentences)
        embedding_matrix = None

        for i in range(0, N, BATCH_SIZE):
            batch_sentences = sentences[i:i+BATCH_SIZE]
            embeddings = embedding_function(batch_sentences)

            # Convert embeddings list of lists to numpy array
            batch_embedding_matrix = np.array(embeddings)

            # Append the batch embedding matrix to the main embedding matrix
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        return similarity_matrix

    def _calculate_reward(self, matrix, start, end):
        sub_matrix = matrix[start:end+1, start:end+1]
        return np.sum(sub_matrix)

    def _optimal_segmentation(self, matrix, max_cluster_size, window_size=3):
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value  # Normalize the matrix
        np.fill_diagonal(matrix, 0)  # Set diagonal to 1 to avoid trivial solutions

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    # local_density = calculate_local_density(matrix, i, window_size)
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    # Adjust reward based on local density
                    adjusted_reward = reward
                    if i - size >= 0:
                        adjusted_reward += dp[i - size]
                    if adjusted_reward > dp[i]:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters
        
    def split_text(self, text: str) -> List[str]:
        sentences = self.splitter.split_text(text)

        similarity_matrix = self._get_similarity_matrix(self.embedding_function, sentences)

        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)

        docs = [' '.join(sentences[start:end+1]) for start, end in clusters]

        return docs

# --- LLM Semantic Chunker ---
class AnthropicClient:
    def __init__(self, model_name, api_key=None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages
            )
            return message.content[0].text
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e
        
class OpenAIClient:
    def __init__(self, model_name, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        try:
            gpt_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages

            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=gpt_messages,
                temperature=temperature
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e


class LLMSemanticChunker(BaseChunker):
    """
    LLMSemanticChunker is a class designed to split text into thematically consistent sections based on suggestions from a Language Model (LLM).
    Users can choose between OpenAI and Anthropic as the LLM provider.

    Args:
        organisation (str): The LLM provider to use. Options are "openai" (default) or "anthropic".
        api_key (str, optional): The API key for the chosen LLM provider. If not provided, the default environment key will be used.
        model_name (str, optional): The specific model to use. Defaults to "gpt-4o" for OpenAI and "claude-3-5-sonnet-20240620" for Anthropic. 
                                    Users can specify a different model by providing this argument.
    """
    def __init__(self, organisation:str="openai", api_key:str=None, model_name:str=None, length_function=len):
        if organisation == "openai":
            if model_name is None:
                model_name = "gpt-4o"
            self.client = OpenAIClient(model_name, api_key=api_key)
        elif organisation == "anthropic":
            if model_name is None:
                model_name = "claude-3-5-sonnet-20240620"
            self.client = AnthropicClient(model_name, api_key=api_key)
        else:
            raise ValueError("Invalid organisation. Please choose either 'openai' or 'anthropic'.")

        self.splitter = RecursiveTokenChunker(
            chunk_size=50,
            chunk_overlap=0,
            length_function=length_function
            )

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                    "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER."
                    "Your response should be in the form: 'split_after: 3, 5'."
                )
            },
            {
                "role": "user", 
                "content": (
                    "CHUNKED_TEXT: " + chunked_input + "\n\n"
                    "Respond only with the IDs of the chunks where you believe a split should occur. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: " + str(current_chunk)+"." + (f"\n\nThe previous response of {invalid_response} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again." if invalid_response else "")
                )
            },
        ]
        return messages

    def split_text(self, text):
        import re

        chunks = self.splitter.split_text(text)

        split_indices = []

        short_cut = len(split_indices) > 0

        from tqdm import tqdm

        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            while True and not short_cut:
                if current_chunk >= len(chunks) - 4:
                    break

                token_count = 0

                chunked_input = ''

                for i in range(current_chunk, len(chunks)):
                    token_count += self.splitter._length_function(chunks[i])
                    chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
                    if token_count > 800:
                        break

                messages = self.get_prompt(chunked_input, current_chunk)
                while True:
                    result_string = self.client.create_message(messages[0]['content'], messages[1:], max_tokens=200, temperature=0.2)
                    # Use regular expression to find all numbers in the string
                    split_after_line = [line for line in result_string.split('\n') if 'split_after:' in line][0]
                    numbers = re.findall(r'\d+', split_after_line)
                    # Convert the found numbers to integers
                    numbers = list(map(int, numbers))

                    # print(numbers)

                    # Check if the numbers are in ascending order and are equal to or larger than current_chunk
                    if not (numbers != sorted(numbers) or any(number < current_chunk for number in numbers)):
                        break
                    else:
                        messages = self.get_prompt(chunked_input, current_chunk, numbers)
                        print("Response: ", result_string)
                        print("Invalid response. Please try again.")

                split_indices.extend(numbers)

                current_chunk = numbers[-1]

                if len(numbers) == 0:
                    break

                pbar.update(current_chunk - pbar.n)

        pbar.close()

        chunks_to_split_after = [i - 1 for i in split_indices]

        docs = []
        current_chunk = ''
        for i, chunk in enumerate(chunks):
            current_chunk += chunk + ' '
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            docs.append(current_chunk.strip())

        return docs

# --- Chunker Factory ---
def get_chunker(strategy: str = "kamradt_modified", rag_agent=None, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Factory function to get the specified chunker instance.
    """
    if rag_agent:
        embed_func = lambda texts: rag_agent.embeddings.embed_documents(texts)
        len_func = len # can be replaced with a token counter
    
    if strategy == "kamradt_modified":
        if not rag_agent:
            raise ValueError("Kamradt chunker requires the rag_agent for its embedding function.")
        return KamradtModifiedChunker(embedding_function=embed_func, length_function=len_func, avg_chunk_size=chunk_size)
    
    elif strategy == "recursive_character":
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif strategy == "fixed_token":
        return FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif strategy == "cluster_semantic":
        if not rag_agent:
            raise ValueError("ClusterSemanticChunker requires the rag_agent for its embedding function.")
        return ClusterSemanticChunker(embedding_function=embed_func, length_function=len_func, max_chunk_size=chunk_size)

    elif strategy == "llm_semantic":
        return LLMSemanticChunker(length_function=len_func)

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")