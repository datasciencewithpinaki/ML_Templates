myname = 'Pinaki'
surname = 'Brahma'

name_l = ['Pinaki', 'Arijita', 'Parameeta', 'Aritra']
surname_l = ['Brahma', 'Ray', 'Brahma', 'Ray']

new_names_l = ['Sambit', 'Ujjal', 'Jagat', 'Suchi', 'Rohit']

full_name_l = name_l + new_names_l

print(f"all names are: {full_name_l}")

name_grtr6 = []
while len(full_name_l)!=0:
    n = full_name_l.pop()
    print(f"popped name: {n}")
    if len(n)>6:
        name_grtr6 = name_grtr6 + [n]

print(name_grtr6)
