
# **Example 1: Using recursion**
def change_base_1(x, base):
    if x < base:
        return str(x)
    return str(base) + change_base_1(x // base, base)

# **Example 2: Using a loop**

def change_base_2(x, base):
    result = ''
    while x > 0:
        result = str(x % base) + result
        x //= base
    return result or '0'

# **Example 3: Using the `int` function and string concatenation**

def change_base_3(x, base):
    result = ''
    while x > 0:
        result = str(int(x % base)) + result
        x //= base
    return result or '0'
# print('exemple 3, resultat: ', change_base(10,2), change_base(0,2))
# **Example 4: Using a list comprehension**

def change_base_4(x, base):
    return ''.join(str(digit) for digit in convert_to_digits(x, base))

def convert_to_digits(x, base):
    digits = []
    while x > 0:
        digits.append(int(x % base))
        x //= base
    return digits[::-1]
# print('exemple 4, resultat: ', change_base(10,2), change_base(0,2))
# **Example 5: Using a generator expression**

def change_base_5(x, base):
    return ''.join(str(digit) for digit in (x // base ** i % base for i in range(len(str(x)) - 1, -1, -1)))

# print('exemple 5, resultat: ', change_base(10,2), change_base(0,2))
# **Example 6: Using the `bin` function and string manipulation**

def change_base_6(x, base):
    bin_str = bin(x)[2:]  # remove '0b' prefix
    return ''.join(str(int(digit, 2) % base) for digit in bin_str[::-1])
# print('exemple 6, resultat: ', change_base(10,2), change_base(0,2))
# **Example 7: Using a dictionary to map digits**

def change_base_7(x, base):
    digit_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    result = ''
    while x > 0:
        digit = int(x % base)
        result = str(digit_map[digit]) + result
        x //= base
    return result or '0'

# print('exemple 7, resultat: ', change_base(10,2), change_base(0,2))
# **Example 8: Using a custom class to convert digits**

class DigitConverter:
    def __init__(self, base):
        self.base = base

    def convert(self, x):
        result = ''
        while x > 0:
            digit = int(x % self.base)
            result = str(digit) + result
            x //= self.base
        return result or '0'

def change_base_8(x, base):
    converter = DigitConverter(base)
    return converter.convert(x)
# print('exemple 8 , resultat: ', change_base(10,2), change_base(0,2))

# **Example 9: Using a lambda function and map**

def change_base_9(x, base):
    return ''.join(map(str, (x // base ** i % base for i in range(len(str(x)) - 1, -1, -1))))

# print('exemple 9, resultat: ', change_base(10,2), change_base(0,2))


# def change_base_10(x, base):
#     result = ''
#     while x > 0:
#         result = str(int(x % base)) + result
#         x //= base
#     return format(result or '0', 'b').translate(str.maketrans('01', '0123456789'))


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