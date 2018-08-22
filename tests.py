w = 3
v = 4
x = 1
y = 1

lr = 0.1

inc = 0

res = 0

while res < 0:
    inc += 1
    res = (w+lr*inc)*x+(v+lr*inc)*y+(-10+lr*inc)
    print(res)
    print(inc)