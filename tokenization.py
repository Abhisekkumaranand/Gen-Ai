import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("vocab Size", encoder.n_vocab)

tokens = encoder.encode("Hello world")
print(tokens)

print(encoder.decode(tokens))

print(len(tokens))