def split_message(text, max_length=2000):
    """
    Splits a string into chunks of max_length, not breaking words.
    Returns a list of message chunks.
    """
    chunks = []
    while len(text) > max_length:
        i = max_length - 1
        while i > 0 and text[i] != " ":
            i -= 1
        if i == 0:
            i = max_length  # fallback: break at max_length
        chunks.append(text[:i])
        text = text[i:].lstrip()
    if text:
        chunks.append(text)
    return chunks
