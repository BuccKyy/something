''''menu = {
    "t": 1.26,
    "z": 3.29,
    "r": 4.14,
    "w": 3.12
}
total = 0.0

print("Menu: ")
for item, price in menu.items():
    print(f"{item}: ${price:2f}")

while True:
    order = input('you can order here')

    if order == 'done':
        break
    if order in menu:
        total += menu[order]
        print(f"{order} for ${menu[order]: 2f}")
    else:
        print("don't have in menu")
print(f"Total ${total:.2f}")'''

'''my_tuple = ['a','a', 'b', 'c', 'd', 'e', 'h']
print(my_tuple.count('a'))
x = my_tuple.pop(1)
y = my_tuple.remove('b')
print(x)
print(my_tuple)'''

'''my_set ={1, 2, 3, 1, 2}
print(my_set)'''

'''from collections import Counter

a = 'aaaaabbbbccc'
my_counter = Counter(a)
print(my_counter)
print(my_counter.most_common(3))  # trả ra phần tử phổ biến có trong hàm  '''


'''from collections import namedtuple

Point = namedtuple('Point', 'x,y')
z = Point(1, -4)
print(z.x, z.y)'''

'''from collections import OrderedDict

# Tạo một OrderedDict
ordered_dict = OrderedDict()

# Thêm các cặp khóa-giá trị
ordered_dict['apple'] = 5
ordered_dict['banana'] = 3
ordered_dict['cherry'] = 8
ordered_dict['date'] = 2

# In toàn bộ OrderedDict, giữ nguyên thứ tự
for key, value in ordered_dict.items():
    print(key, value)'''


'''from collections import defaultdict

# Tạo một defaultdict với giá trị mặc định là 0
default_dict = defaultdict(int)

# Thêm các cặp khóa-giá trị
default_dict['apple'] = 5
default_dict['banana'] = 3

# Truy cập một khóa không tồn tại
value = default_dict['cherry']
print(value)  # Kết quả: 0, vì giá trị mặc định là 0'''


''''from collections import deque

# Tạo một deque
my_deque = deque()

# Thêm phần tử vào cuối danh sách
my_deque.append(1)
my_deque.append(2)

# Thêm phần tử vào đầu danh sách
my_deque.appendleft(0)

# In toàn bộ danh sách
print(my_deque)  # Kết quả: deque([0, 1, 2])

# Loại bỏ phần tử cuối cùng
my_deque.pop()

# Loại bỏ phần tử đầu tiên
my_deque.popleft()

# In toàn bộ danh sách sau khi loại bỏ
print(my_deque)  # Kết quả: deque([1])'''


'''import matplotlib.pyplot as plt

# Dữ liệu mẫu
x = [1, 2, 3, 4, 5]
y = [10, 15, 13, 18, 20]

# Tạo biểu đồ đường
plt.plot(x, y)

# Đặt tên cho trục x và trục y
plt.xlabel('Trục X')
plt.ylabel('Trục Y')

# Đặt tiêu đề cho biểu đồ
plt.title('Biểu đồ đường đơn giản')

# Hiển thị biểu đồ
plt.show()'''

'''import numpy as np


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[1:]) + self.w_[0]


def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.net_input(X) >= 0.0, 1, -1)'''

'''from itertools import product

a = [1, 2, 3]

b = ['a', 'b', 'c']

products = list(product(a, b))
print(products)'''

# [(1, 'a'), (1, 'b'), (1, 'c'), (2, 'a'), (2, 'b'), (2, 'c'), (3, 'a'), (3, 'b'), (3, 'c')]


'''from itertools import permutations

a = [1, 2, 3]
perm = permutations(a, 3)
print(list(perm))
#[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]'''


'''from itertools import combinations, combinations_with_replacement
a = [1, 2, 3, 4]
comb = combinations(a,2)
print(list(comb))
comb_wr = combinations_with_replacement(a, 2)
print(list(comb_wr))
#[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
#[(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]'''

'''from itertools import accumulate
import operator
a = [1, 2, 3, 4]
#acc = accumulate(a,) #1
acc = accumulate(a, func=operator.mul) #2
print(a)
print(list(acc))
#1[1, 2, 3, 4]
#1[1, 3, 6, 10] # cong tang dan don vi
#2[1, 2, 3, 4] # nhan tang dan don vi
#2[1, 2, 6, 24]'''

'''from itertools import groupby

def smaller_than_3(x):
    return x < 3

a = [1, 2, 3, 4]
g_o = groupby(a, key=smaller_than_3)    #True [1, 2]
for key, value in g_o:                  #False [3, 4]
    print(key, list(value))'''

'''from itertools import groupby

persons = [{'name': 'Tim', 'age': 25}, {'name': 'Dan', 'age': '25'},
           {'name': 'Lisa', 'age': 27}, {'name': 'Claire', 'age': 28}]

g_o = groupby(persons, key=lambda x: x['age'])
for key, value in g_o:
    print(key, list(value))

#25 [{'name': 'Tim', 'age': 25}]
#25 [{'name': 'Dan', 'age': '25'}]
#27 [{'name': 'Lisa', 'age': 27}]
#28 [{'name': 'Claire', 'age': 28}]'''

'''from itertools import count, cycle, repeat
a = [1, 2, 3]
for i in count(10):
    print(i)
    if i == 15:
        break

for i in repeat(1, 4):
    print(i)

for i in cycle(a):
    print(i)'''


'''add10 = lambda x: x + 10
print(add10(5))

def add10_func(x):
    return x + 10


mult = lambda x, y: x*y
print(mult(2,7))'''

'''points2D = [(1, 2), (15, 1), (5, -1), (10, 4)]

def s_b_y(x):
    return x[1]

point2D_sorted = sorted(points2D, key=lambda x: x[1])

print(points2D)
print(point2D_sorted)'''


# lambda arguments: expression
# map(func, seq)

a = [1, 2, 3, 4, 5]
b = map(lambda x: x*2, a)
print(list(b))

c = [x*2 for x in a]
print(c)