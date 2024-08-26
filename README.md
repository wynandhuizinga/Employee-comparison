# Retrieval Augmented Generation
### *Comparing your closest colleagues*

This repository is intended to evaluate an LLM's capability of comparing colleagues with each other to determine similarities. 

In summary, the script does 3 things:
1) It transforms a number of text entries into embeddings (building a vector-database).
2) 1 entry is compared to all the other ones to determine similarity ( = vector 'database' search). The output gets sorted.
3) The subject and its 'neirest neighbor' are send to a GPT with a prompt to request for 'discussion topics' ( = retrieval augmented generation).

The beauty of it is, that the actual data entered can be structured, but unlike traditional database querying, this approach is forgiving to 'less structured' data. Successful matching does heavily depend on which LLM is used.

### Result
See below the outcome of succesful execution of this script. Comments in line of the script provide a more detailed explanation. 
![Employee Comparison Demo](https://github.com/wynandhuizinga/Employee-comparison/blob/main/Employee-comparison-demo.png)

This solution relies on Ollama with Meta's Llama 3 model hosted on a localhost. However, if adopted in enterprise architecture, it can also be adjusted and connected to more sophisticated models such as OpenAI's chatgpt 4. 

### Acknowledgement
Thanks to NeuralNine for providing accessible coarse contents on which this repository is inspired.
