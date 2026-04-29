import pandas as pd 
import torch

class CharTokeniser():
    """Tokeinser class for deriving char-to-id and id-to-char mappings used in encoding and decoding.
    
    Attributes:
        stress_markers (set): the two stress markers used in Spanish IPA.
        char_to_id (dict): character to integer mapping used for encoding.
        id_to_char (dict): integer to character mapping used for decoding.
        vocab (list): all unique graphemes and phonemes.
        vocab_size (int): number of unique tokens (always 21 if derived from train set, as it should be).
        max_len (int): length of longest train IPA sequence + 3 for special tokens. 
    """
    def __init__(self, df: pd.DataFrame):
        """ 
        PAD: 0
        BOS: 1
        EOS: 2
        UNK: 3

        Args:
            df: the training datset from which to derive the vocabulary
        """

        word_chars = set(char for word in df['word'] for char in word)

        self.stress_markers = {'ˌ', 'ˈ'}

        phonemes = set()
        for ipa_seq in df['ipa']:
            for phoneme in ipa_seq.split():
                if phoneme[0] in self.stress_markers:
                    phonemes.add(phoneme[0])
                    phonemes.add(phoneme[1:])
                else:
                    phonemes.add(phoneme)

        self.vocab = list(sorted(word_chars | phonemes))
        self.char_to_id = {char: i+4 for i, char in enumerate(self.vocab)}
        self.id_to_char = {i+4: char for i, char in enumerate(self.vocab)}
        self.vocab_size = int(len(self.char_to_id) + 4)
        self.max_len = int(df['ipa'].apply(lambda x: len(self._ipa_to_units(x)) + 2)).max()

    def _ipa_to_units(self, sequence: str) -> list:
        """Split a space-separated IPA sequence into phonemes. This method is necessary in order to treat the stress markers as their own unit.

        Args:
            sequence (str): a space-separated IPA sequence like p ɾ e k a ˈb i d o

        Returns:
            List of phonemic units, not including whitespace. For example, t͡ʃ is treated as its own unit instead of multiple tokens.
        """
        units = []
        for token in sequence.split():
            if token[0] in self.stress_markers:
                # Treat stress marker as separate unit
                units.append(token[0])
                units.append(token[1:])
            else:
                units.append(token)

        return units

    def encode(self, inputs: str | list, is_ipa: bool=False) -> torch.Tensor:
        """Encode a sequence of graphemes or IPA as integers. If a list of inputs is provided, they are padded to the maximum length sequence.
        
        Args:
            inputs (str | list): a space-separated sequence of graphemes or IPA.
            is_ipa (bool): True if input is IPA, False if input is graphemes.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        tokenised_list = []
        for sequence in inputs:
            if is_ipa:
                units = self._ipa_to_units(sequence)
            else:
                units = list(sequence)
            tokens = [1] + [self.char_to_id.get(u, 3) for u in units] + [2]
            tokenised_list.append(tokens)

        max_len = max(len(t) for t in tokenised_list)
        padded = [t + [0] * (max_len - len(t)) for t in tokenised_list]

        return torch.tensor(padded).long()
    
    def decode(self, tokens: torch.Tensor):
        """Decode a tokenised integers into IPA phonemes. """
        if torch.is_tensor(tokens):
            tokens = tokens.flatten().tolist()

        # Get graphemes/IPA but don't encode special tokens
        units = [self.id_to_char.get(t, '') for t in tokens if t > 3]

        result = []
        i = 0
        while i < len(units):
            if units[i] in self.stress_markers and i + 1 < len(units):
                # If we hit a stress marker, combine it with the next token
                result.append(units[i] + units[i+1])
                i += 2 
            else:
                result.append(units[i])
                i += 1
            
        return ' '.join(result)