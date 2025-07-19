from alchemy.models import AlchemyCode, AlchemyCodeBinary

def code_to_binary_int(char: str) -> int:
    if char.isdigit():
        return int(char)
    elif char.isalpha() and char.isupper():
        return ord(char) - ord('A') + 10
    elif char.isalpha() and char.islower():
        return ord(char) - ord('a') + 36
    elif char in "!?":
        return 62 if char == '?' else 63
    else:
        raise ValueError(f"Invalid character in alchemy code: {char}")

def binary_integer_to_code(binary_integer: int) -> str:
    if binary_integer < 10:
        return str(binary_integer)
    elif 10 <= binary_integer < 36:
        return chr(binary_integer - 10 + ord('A'))
    elif 36 <= binary_integer < 62:
        return chr(binary_integer - 36 + ord('a'))
    elif binary_integer == 62:
        return '?'
    elif binary_integer == 63:
        return '!'
    else:
        raise ValueError(f"Invalid binary integer for alchemy code: {binary_integer}")

def code_to_binary(code: AlchemyCode) -> AlchemyCodeBinary:
    return [code_to_binary_int(char) for char in code]

def binary_to_code(binary: AlchemyCodeBinary) -> AlchemyCode:
    return ''.join(binary_integer_to_code(b) for b in binary)

def alchemy_and(code_1: AlchemyCode, code_2: AlchemyCode) -> AlchemyCode:
    print(code_to_binary(code_1), code_to_binary(code_2))
    return binary_to_code([(b1 & b2) for b1, b2 in zip(code_to_binary(code_1), code_to_binary(code_2))])

def alchemy_or(code_1: AlchemyCode, code_2: AlchemyCode) -> AlchemyCode:
    return binary_to_code([(b1 | b2) for b1, b2 in zip(code_to_binary(code_1), code_to_binary(code_2))])