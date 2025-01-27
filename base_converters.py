from human_eval.evaluation import evaluate_functional_correctness
 
def change_base_s1(x, base):
    if x < base:
        return str(x)
    return str(base) + change_base_s1(x // base, base)


def change_base_s2(x, base):
    result = ''
    while x > 0:
        result = str(x % base) + result
        x //= base
    return result or '0'

def change_base_s3(x, base):
    result = ''
    while x > 0:
        result = str(int(x % base)) + result
        x //= base
    return result or '0'

def change_base_s4(x, base):
    return ''.join(str(digit) for digit in convert_to_digits(x, base))

def convert_to_digits(x, base):
    digits = []
    while x > 0:
        digits.append(int(x % base))
        x //= base
    return digits[::-1]

def change_base_s5(x, base):
    return ''.join(str(digit) for digit in (x // base ** i % base for i in range(len(str(x)) - 1, -1, -1)))

def change_base_s6(x, base):
    bin_str = bin(x)[2:]  # remove '0b' prefix
    return ''.join(str(int(digit, 2) % base) for digit in bin_str[::-1])

def change_base_s7(x, base):
    digit_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    result = ''
    while x > 0:
        digit = int(x % base)
        result = str(digit_map[digit]) + result
        x //= base
    return result or '0'

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

def change_base_s8(x, base):
    converter = DigitConverter(base)
    return converter.convert(x)

def change_base_s9(x, base):
   return ''.join(map(str, (x // base ** i % base for i in range(len(str(x)) - 1, -1, -1))))

def change_base_s10(x, base):
    result = ''
    while x > 0:
        result = str(int(x % base)) + result
        x //= base
    return format(result or '0', 'b').translate(str.maketrans('01', '0123456789'))


solutions = {
    "0": change_base_s1,
    "1": change_base_s2,
    "2": change_base_s3,
    "3": change_base_s4,
    "4": change_base_s5,
    "5": change_base_s6,
    "6": change_base_s7,
    "7": change_base_s8,
    "8": change_base_s9,
    "9": change_base_s10
}