"""
Author: Adriel Kuek
Date Created: 09 Oct 2021
Version: 0.1
Email: adrielkuek@gmail.com
Status: Devlopment

Description:
CLIP retrieval is based on OpenAI's CLIP model for Image and Text retrieval
through contrastive language-Image Pretraining.


"""
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from PIL import Image

import torch
import clip
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

# NLP Processing Auxillary Functions
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

class CLIP(object):
    def __init__(self, model_dir, tensor_file, img_filename_pickle):
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP base model - Using ViT-B/32 model (Offline model)
        print(f'LOADING CLIP PRETRAINED . . .')
        self.model, self.preprocess = clip.load(model_dir, device=self.device, offline=True)

        # Load dataset tensor
        self.dataset_tensor = torch.load(tensor_file)

        # Load Image Filenames
        opened_pickle = open(img_filename_pickle, "rb")
        self.img_filename = pickle.load(opened_pickle)
        opened_pickle.close()

    def kNN_retrieval(self, Topk, input, is_text=False):
        
        image_results_list = []

        # Process image input query
        if is_text is False:
            imageQuery = self.preprocess(input).unsqueeze(0).to(self.device)

            # Perform image feature extraction
            with torch.no_grad():
                image_encode_time = time.time()
                image_features = self.model.encode_image(imageQuery)
                print(f'Image Encode Time: {time.time() - image_encode_time} secs')
            
            # Feature Normalisation
            image_features /= image_features.norm(dim=1, keepdim=True)

            # Compute NN similarity
            similarities_images = self.dataset_tensor @ image_features.T
            # Compute for TopK
            img_value, img_retrieved_idx = similarities_images.topk(Topk, dim=0)

            for id in img_retrieved_idx:
                image_results_list.append(self.img_filename[id])

        else:   # process text input query

            # Text Preprocessing
            # We append text-prompts for CLIP encode in front to
            # improve classification performance
            tokenizer = SimpleTokenizer()
            text_tokens = [tokenizer.encode("This is a picture of " + input)]

            # Handle abnormal inputs
            # Check extracted text sentence length - CLIP tokenizer has a upper bound limit
            # Check for return NULL
            if len(text_tokens[0]) == 0:
                text_tokens = ['NULL']

            # Chop at string length 70
            if len(text_tokens[0]) > 70:
                text_tokens = [text_tokens[0][:70]]

            # Define start and end stream
            text_input = torch.zeros(len(text_tokens), self.model.context_length, dtype=torch.long)
            sot_token = tokenizer.encoder['<|startoftext|>']
            eot_token = tokenizer.encoder['<|endoftext|>']

            # Parse start and end Tags to input text
            for i, tokens in enumerate(text_tokens):
                tokens = [sot_token] + tokens + [eot_token]
                text_input[i, :len(tokens)] = torch.tensor(tokens)

            # Transfer to GPU to use CUDA
            text_input = text_input.to(self.device)

            # Perform Text encoding
            with torch.no_grad():
                text_encode_time = time.time()
                text_features = self.model.encode_text(text_input)
                print(f'Text Encode Time: {time.time() - text_encode_time} secs')

            # Normalisation Text Features
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute Similarities
            similarities_images = self.dataset_tensor @ text_features.T
            # Compute for TopK
            img_value, img_retrieved_idx = similarities_images.topk(Topk, dim=0)

            for id in img_retrieved_idx:
                image_results_list.append(self.img_filename[id])
        
        return image_results_list

# For Testing of class object
def main():

    model_dir = '/home/user/Adriel/MTech-IS/GradCert_PatternRecognition/PRS_PractiseModule/Imantics.io/ViT-B-32.pt'
    input_filename_pickle = '/home/user/Adriel/MTech-IS/GradCert_PatternRecognition/PRS_PractiseModule/Imantics.io/OurDataset_Tensors/image_filenames_OurDataset.pkl'
    dataset_tensor = '/home/user/Adriel/MTech-IS/GradCert_PatternRecognition/PRS_PractiseModule/Imantics.io/OurDataset_Tensors/image_embeddings_OurDataset.pt'
    
    imageResults_list = []

    clip = CLIP(model_dir, dataset_tensor, input_filename_pickle)

    # Test with sample image
    # Retrieve Test Image Query
    imgSample_filepath = '/media/user/New Volume/TINKERMAN/OurDataset/PRS_CombinedDataset/train/14311265_10154545979703896_614300630827326582_o.jpg'
    imageSample = Image.open(imgSample_filepath)

    # # Display Here
    # plt.figure(figsize=(15, 15))
    # plt.imshow(imageSample)

    imageResults_list = clip.kNN_retrieval(10, imageSample, is_text=False)
    print(imageResults_list)

    # Text with sample Text query
    sampleText = 'cats'
    imageResults_list = clip.kNN_retrieval(10, sampleText, is_text=True)
    print(imageResults_list)

if __name__ == "__main__":
    main()


