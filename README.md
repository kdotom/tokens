Run the transform.py file to generate text using an auto-regressive method with embeddings.

Run the tensors.py file to generate the values of the safetensors from huggingface.

Todo: Determine order of calculation for each subnetwork. Maybe start from the end and determine feeds in recursively?
Oh. I guess it (Llama 3) is just passing one layer to the next, nothing complicated. See this page:
https://towardsdatascience.com/deep-dive-into-llama-3-by-hand-%EF%B8%8F-6c6b23dc92b2
