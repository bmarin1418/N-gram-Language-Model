This is a Python class for an n-gram character level language model. It also implements stupid backoff, which recursively reduces the markov order if an example at the highest n-gram order isn't found. This means that we keep information for all the n-grams and I wrote this for readability over performance (better learning for me) so be warned!

Example Usage

```python
# Get the language model ready
n_gram = 5
lang_model = Ngram(n_gram)
lang_model.train("lots_of_enlgish_text.txt")

correct_preds = 0
total_preds = 0
with open("lots_of_different_english.txt", 'r') as fd:
    for line in fd.readlines():
        for char in line:
            highest_prob, higest_prob_char = max([[lang_model.prob(char), char] for char in string.printable])
            if char == higest_prob_char:
                correct_preds += 1
            total_preds += 1
            lang_model.read(char) # Remember to tell the language model what you just saw
print("Five Gram with Stupid Backoff Accuracy: {:.4}%".format(100 * correct_preds/total_preds))
```
