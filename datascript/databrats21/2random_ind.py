import random

train=r'txt\train.txt'
val=r'txt\val.txt'
test=r'txt\test.txt'

random_train=r'txt\rtrain.txt'
random_val=r'txt\rval.txt'
random_test=r'txt\rtest.txt'

names = None

with open(train, 'r') as f:
    names = f.readlines()

names = [name.strip() for name in names]
random.shuffle(names)
with open(random_train, 'w') as f:
    for name in names:
        f.write(name + '\n')

names = None
with open(val, 'r') as f:
    names = f.readlines()
names = [name.strip() for name in names]
random.shuffle(names)
with open(random_val, 'w') as f:
    for name in names:
        f.write(name + '\n')

names = None

with open(test, 'r') as f:
    names = f.readlines()
names = [name.strip() for name in names]
random.shuffle(names)
with open(random_test, 'w') as f:
    for name in names:
        f.write(name + '\n')

print("！")