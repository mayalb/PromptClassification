
def change_base_1(x, base):
    return ''.join(str(int(digit, 10)) for digit in format(x, 'b').zfill(base))

def change_base_2(x, base):
    return bin(x)[2:].zfill(base)

def change_base_3(x, base):
    if x < base:
        return str(x)
    else:
        return change_base_3(x // base, base) + str(x % base)

def change_base_4(x, base):
    return ''.join([str((x // (base ** i)) % base) for i in range(0, len(str(x)))])

def change_base_5(x, base):
    d = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    return ''.join([str((x // (base ** i)) % base) if (x // (base ** i)) % base < 10 else d[x // (base ** i) % 16] for i in range(0, len(str(x)))])

change_base_6 = lambda x, base: ''.join([str((x // (base ** i)) % base) if (x // (base ** i)) % base < 10 else chr(ord('a') + (x // (base ** i)) % 16 - 10) for i in range(0, len(str(x)))])

def change_base_7(x, base):
    return ''.join(map(lambda y: str(y) if y < 10 else chr(ord('a') + y - 10), [int(digit) for digit in format(x, 'b').zfill(base)]))

def change_base_8(x, base):
    d = {i: str(i) if i < 10 else chr(ord('a') + i - 10) for i in range(10)}
    return ''.join([d[int(digit)] for digit in format(x, 'b').zfill(base)])

def change_base_9(x, base):
    for i in range(len(str(x))):
        yield str((x // (base ** i)) % base) if (x // (base ** i)) % base < 10 else chr(ord('a') + (x // (base ** i)) % 16 - 10)

def change_base_10(x, base):
    result = ''
    for i in range(len(str(x))):
        digit = (x // (base ** i)) % base
        if digit < 10:
            result += str(digit)
        else:
            result += chr(ord('a') + digit - 10)
    return result

change_base_functions = [
    change_base_1,
    change_base_2,
    change_base_3,
    change_base_4,
    change_base_5,
    change_base_6,
    change_base_7,
    change_base_8,
    change_base_9,
    change_base_10

]

# Input values
x = int(input("Enter the value of x: "))
base = int(input("Enter the base: "))

# Loop through each function and print the result
for i, func in enumerate(change_base_functions, 1):
    try:
        result = func(x, base)
        print(f"Example {i}, Result: {result}")
    except Exception as e:
        print(f"Example {i}, Error: {e}")