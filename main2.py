print(max(map(len, open('24_demo.txt').readline().replace('Z', ' ').replace('Y', ' ').split())))

file = open("24_demo.txt", "r")
str = file.read()
max = 0
count = 0
for i in str:
    if i == "X":
        count += 1
    else:
        if max < count:
            max = count
        count = 0
print(max)


for n in range(1, 50):
    if "X" * n in str:
        print(n)
