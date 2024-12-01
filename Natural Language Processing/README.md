# Language Models

## What is a Language Model?

Given an initial context, a language model estimates the probability distribution over all possible next tokens. Let's look at one example with tokens being full words.

$$
    \mathbb{P}(token\;|\;\text{My} \cap \text{car} \cap \text{is}) = 
        \begin{cases} 
             20\% & \text{if } token = \text{broken} \\
             17\% & \text{if } token = \text{fast} \\
             8\% & \text{if } token = \text{slow} \\
             ...
        \end{cases}
$$

In Bigram models, for instance, this context window has the size of a single token, which makes it really not that good. Let's see what the previous example would look like with a 1-word context window.

$$
    \mathbb{P}(token\;|\;\text{is}) = 
        \begin{cases} 
             13\% & \text{if } token = \text{happy} \\
             10\% & \text{if } token = \text{sad} \\
             7\% & \text{if } token = \text{changing} \\
             ...
        \end{cases}
$$

Note how this difference in context completely changes the distributions for the next token.

## What are tokens?

Language models work by splitting strings (sentences, paragraphs, entire books) into small elements, called tokens. A sequence of such tokens, then, make up the entirety of the string, and a myriad of probabilistic approaches can be used to model their distributions.

In early NLP research, such tokens were taken as full words, like in the examples above (e.g. $token_1 = \text{My}$), but this approach quickly becomes a problem when one realizes that separating words is not an objective task across all languages (e.g. is Orangensaft 1 or 2 words?). There is also the likelihood that a new word will appear during inference, and the model will have no idea how to treat it. Thus, nowadays, it is much more common to use subwords, which are groups of characters that tend to appear together in the corpus (e.g. th, ish, ...).