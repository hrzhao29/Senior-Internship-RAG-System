Outline:
1. main.py contains the RAG system
2. test_eval evaluates the system using Deepeval and an external dataset. 

WixQA:
- Downloaded dataset from https://huggingface.co/datasets/Wix/WixQA

Random Data Generation:
1. generate_docs.py gets source context from a webpage
2. generate_eval_data.py uses Deepeval's golden synthesizer to create golden's (inputs + expected outputs from a context chunk) from the source context gathered in 1