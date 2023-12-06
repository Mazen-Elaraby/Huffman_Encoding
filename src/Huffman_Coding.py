import os
import numpy as np
from file_io import *
from heapq import heapify, heappop, heappush

class Node:
    def __init__(self, ch, freq, left=None, right=None):
        self.ch = ch
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        #overloading the less than operator for heapq to be able to maintain the priority queue
        return self.freq < other.freq

class huffman_coding:
    def __init__(self, path):
        self.path = path
        self.text = read_from_txt_file(path)
        self.root = None

        #File Characteristics
        self.freq_dict = None # a dictionary with symbol frequencies
        self.prob_dict = None # a dictionary with symbol probabilities 
        self.info_dict = None # a dictionary with symbol information
        self.src_entropy = None

        #Codebook characteristics
        self.codebook = {}
        self.code_avg_len = None #Code Average Length
        self.code_eff = None #Code Efficiency

        self.compression_ratio = None

    def get_freq_dict(self):
        """
        Calculates and returns the frequency of each character in the text.

        Returns:
            dict: A dictionary where the keys are characters and values are the frequencies 
                of these characters in the text.
        """

        self.freq_dict = {ch: self.text.count(ch) for ch in set(self.text)}

        return self.freq_dict
    
    def get_prob_dict(self, freq_dict):
        """
        Calculates and returns the probability of each character in the text.

        Parameters:
            freq_dict (dict): A dictionary where keys are characters and values are the frequencies of these
                            characters in the text.

        Returns:
            dict: A dictionary where the keys are characters and the values are the probabilities of these 
                characters in the text.
        """

        tot_no_symbols = sum(freq_dict.values())

        self.prob_dict = {symbol: (freq / tot_no_symbols) for symbol, freq in freq_dict.items()}

        return self.prob_dict

    
    def get_info_dict(self, prob_dict):
        """
        Calculates and returns the information content for each character in the text.

        Parameters:
            prob_dict (dict): A dictionary where keys are characters and values are the probabilities
                            of these characters in the text.

        Returns:
            dict: A dictionary where the keys are characters and the values are the information content
                (in bits) of these characters.
        """

        self.info_dict = {symbol: -np.log2(prob) for symbol, prob in prob_dict.items()}

        return self.info_dict

    def get_src_entropy(self, prob_dict):
        """
        Calculates and returns the source entropy of the text.

        Parameters:
            prob_dict (dict): A dictionary where keys are characters and values are the probabilities of these
                            characters in the text.

        Returns:
            float: The calculated source entropy of the text, representing the average number of bits needed per symbol.
        """
        
        self.src_entropy = -sum(prob * np.log2(prob) for prob in prob_dict.values())

        return self.src_entropy
    
    def get_codebook(self, freq_dict):
        """
        Generates and returns the Huffman codebook for the text.

        The method consists of two internal functions: `get_huffman_tree` and `generate_codebook`. 
        `get_huffman_tree` builds the tree, and `generate_codebook` performs the recursive traversal 
        to create the code for each character. The final codebook is stored in the `codebook` attribute 
        of the class.

        Parameters:
            freq_dict (dict): A dictionary where keys are characters and values are the frequencies of these
                            characters in the text.

        Returns:
            dict: A codebook where keys are characters and values are their corresponding Huffman codes.
        """
        
        def get_huffman_tree():

            pq = [Node(k,v) for k, v in freq_dict.items()] #priority queue
            heapify(pq)

            while (len(pq) > 1):
                left, right = heappop(pq), heappop(pq)
                new_freq = left.freq + right.freq
                heappush(pq, Node(None, new_freq, left, right))

            self.root = pq[0] 

        def generate_codebook(node, curr_code=""):
            #traversing huffman tree recursively to form the codebook
            if (node.ch is not None): #stopping condition: reached leaf node
                self.codebook[node.ch] = curr_code
                return
            
            generate_codebook(node.left, curr_code + "1")
            generate_codebook(node.right, curr_code + "0")

        get_huffman_tree()
        generate_codebook(self.root)

        return self.codebook

    def encode(self):
        """
        Encodes the text using the Huffman codebook and writes the encoded text to a binary file.

        The method assumes that `self.path` holds the path to the original text file and `self.codebook` contains 
        the Huffman codebook. The method `write_to_bin_file` is used to write the encoded text to a binary file. 
        The method updates the `compression_ratio` attribute with the calculated compression ratio.
        """
        #encode text
        encoded_text = "".join([self.codebook[symbol] for symbol in self.text]) 

        #write to binary file
        output_filename = "compressed_" + self.path.split(".")[0] + ".bin" 
        write_to_bin_file(output_filename, encoded_text)

        #calculate compression ratio
        self.compression_ratio = (os.path.getsize(output_filename) / os.path.getsize(self.path)) * 100

    def get_compression_ratio(self):
        """
        Returns:
            float: File Compression Ratio 
        """
        return self.compression_ratio
    
    def get_code_avg_len(self, prob_dict, codebook):
        """
        Calculates and returns the average length of the Huffman codes.

        Parameters:
            prob_dict (dict): A dictionary where keys are characters and values are the probabilities of these
                            characters in the text.
            codebook (dict): A dictionary where keys are characters and values are their corresponding Huffman codes.

        Returns:
            float: The average length of the Huffman codes for the characters in the text.
        """
        
        self.code_avg_len = sum(prob_dict[symbol] * len(codebook[symbol]) for symbol in prob_dict)

        return self.code_avg_len

    def get_code_eff(self, src_entropy, code_avg_len):
        """
        Calculates and returns the coding efficiency of the Huffman encoding using the provided source entropy and average code length.

        Parameters:
            src_entropy (float): The source entropy, representing the theoretical minimum average code length.
            code_avg_len (float): The average length of the Huffman codes for the encoded text.

        Returns:
            float: The calculated coding efficiency of the Huffman coding, reflecting how efficiently the text has been encoded.
        """
        
        self.code_eff = (src_entropy / code_avg_len) * 100

        return self.code_eff

    def decode(self):
        """
        Decodes the encoded text from a binary file and writes the decoded text to a text file.

        The method assumes that `self.root` holds the root of the Huffman tree. The names of the input (encoded) and 
        output (decoded) files are derived from the original text file's name, prefixed with 'compressed_' and 
        'decompressed_', respectively. The methods `read_from_bin_file` and `write_to_txt_file` are used to 
        read the encoded text from a binary file and to write the decoded text to a text file, respectively.
        """
        #read encoded text from binary file
        input_filename = "compressed_" + self.path.split(".")[0] + ".bin" 
        encoded_text = read_from_bin_file(input_filename)
        #decode
        decoded_text = []
        curr_node = self.root 
        for bit in encoded_text:

            curr_node = curr_node.right if bit == '0' else curr_node.left

            if (curr_node.ch is not None):
                decoded_text.append(curr_node.ch)
                curr_node = self.root
        
        decoded_text = ''.join(decoded_text)
 
        #write to text file
        output_filename = "decompressed_" + self.path.split(".")[0] + ".txt"
        write_to_txt_file(output_filename, decoded_text)