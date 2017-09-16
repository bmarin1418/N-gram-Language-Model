import collections

####################################
## Ngram language model class
####################################

class Ngram(object):
####################################
##    Public Functions
#################################### 

    def __init__(self, ngrams):
        self.char_counts = collections.Counter() # For unigram model
        self.total_char_count = 0                # Also for unigram model
        ngrams = int(ngrams)
        assert ngrams > 1, "Mimimum supported n is 2"
        self.n_grams = ngrams
        self.conditonal_counts = [collections.Counter() for _ in range(self.n_grams - 1)] # Keep conditionals for every n. Easy to read, bad for performance though
        self.joint_prob_counts = [collections.Counter() for _ in range(self.n_grams - 1)]
        self.backoff_counts = collections.Counter() # Keep some stats on when we have to backoff from the max n model

    def train(self, filename):
        """Train the model on a text file."""
        self.start()
        for line in open(filename):
            for i, char in enumerate(line):
                self.char_counts[char] += 1
                self.total_char_count += 1
                
                # Record counts for n gram models
                for markov_order_i in range(self.n_grams - 1):
                    if len(self.conditional_chars) >= markov_order_i + 1:
                        self.__initOrAdd(self.conditonal_counts[markov_order_i], self.conditional_chars[-1 * (markov_order_i + 1):], 1)
                        self.__initOrAdd(self.joint_prob_counts[markov_order_i], self.conditional_chars[-1 * (markov_order_i + 1):] + char, 1)

                self.__updateConditionalChars(char)
        self.start()

    def start(self):
        """Reset the state to the initial state."""
        self.conditional_chars = ""

    def read(self, char):
        """Read in character char, updating the state."""
        self.__updateConditionalChars(char)

    def prob(self, char):
        """Return the probability of the next character being char given the
        current state."""

        # Stupid backoff
        for markov_i in range(self.n_grams - 2, -1, -1):
            conditional_str = self.conditional_chars[:markov_i + 1]
            if conditional_str in self.conditonal_counts[markov_i]:
                self.backoff_counts[markov_i + 2] += 1
                return .4**markov_i * self.joint_prob_counts[markov_i][conditional_str + char] / self.conditonal_counts[markov_i][conditional_str]
        self.backoff_counts[1] += 1
        return self.char_counts[char] / self.total_char_count

    def displayModelUsages(self):
        """Show frequency of backing off"""
        total_preds = sum([self.backoff_counts[key] for key in self.backoff_counts])
        for key in self.backoff_counts:
            print("{}-gram usage: {}%".format(int(key), self.backoff_counts[key] * 100/total_preds))

####################################
##    Private Functions
#################################### 

    def __updateConditionalChars(self, char):
        """Update the last chars seen"""
        if len(self.conditional_chars) < self.n_grams - 1:
            self.conditional_chars = self.conditional_chars + char
        else:
            self.conditional_chars = self.conditional_chars[1:] + char

    def __initOrAdd(self, counter, key, increment):
        """Initialize to the increment if key isn't in dict
            or increment if it is"""
        if key in counter:
            counter[key] += increment
        else:
            counter[key] = increment



if __name__ == "__main__":
    import argparse
    import string

    parser = argparse.ArgumentParser()
    parser.add_argument('TRAIN_FILE')
    parser.add_argument('TEST_FILE')
    parser.add_argument('n')
    args = parser.parse_args()

    lang_model = Ngram(args.n)
    print("Training, be patient...")
    lang_model.train(args.TRAIN_FILE)

    # Now see how many times the highest prob char unigram will always predict appears
    correct_preds = 0
    total_preds = 0
    print("Evaluating performance...")
    with open(args.TEST_FILE, 'r') as fd:
        for line in fd.readlines():
            for char in line:
                highest_prob, higest_prob_char = max([[lang_model.prob(char), char] for char in string.printable])
                if char is higest_prob_char:
                    correct_preds += 1
                total_preds += 1
                lang_model.read(char)
    print("Five Gram with Stupid Backoff Accuracy: {:.4}%".format(100 * correct_preds/total_preds))
