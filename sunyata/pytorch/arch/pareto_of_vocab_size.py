"""
What is the best Pareto size of vocabulary in language modeling?
Prior: the distribution of word counts (probability) is zipf (or scale free)
The expected token sequence length of a sentence decreasing with the vocabulary size.
In a language model, the predicting precision decreasing with the vocabulary size.
The total precision of predicting a sentence is the production of precision of each token.
Suppose the token sequence length is $s$, the vocabulary size is $v$. 
Then, for a fixed sentence, $s$ is a decreasing function of $v$.
While the predicting precision $p_i, i \in (1, i, s)$ of a language model with a fixed parameter number,
is a decreasing function of $v$.
What is the best precision $\prod_1^s p_i$? the best vocab_size
"""