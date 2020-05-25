import numpy as np


def get_p_b_cd():
    # you need to implement this method.
    p_b_cd = np.zeros((3, 3, 2), dtype=np.float)
    for line in content[1:]:
        data = list(map(int, line.strip().split('\t')[1:]))
        b = data[1]
        c = data[2]
        d = data[3]
        p_b_cd[b - 1][c - 1][d - 1] += 1

    for c in range(3):
        for d in range(2):
            el_sum = p_b_cd[0][c][d] + p_b_cd[1][c][d] + p_b_cd[2][c][d]
            for b in range(3):
                p_b_cd[b][c][d] /= el_sum

    return p_b_cd


def get_p_a_be():
    # you need to implement this method.
    p_a_be = np.zeros((2, 3, 2), dtype=np.float)
    for line in content[1:]:
        data = list(map(int, line.strip().split('\t')[1:]))
        a = data[0]
        b = data[1]
        e = data[4]
        p_a_be[a - 1][b - 1][e - 1] += 1

    for b in range(3):
        for e in range(2):
            el_sum = p_a_be[0][b][e] + p_a_be[1][b][e]
            for a in range(2):
                p_a_be[a][b][e] /= el_sum

    return p_a_be


# following lines are main function:
data_add = "data//assign2_BNdata.txt"

with open(data_add, 'r') as f:
    content = f.readlines()

# probability distribution of b.
p_b_cd = get_p_b_cd()
for c in range(3):
    for d in range(2):
        for b in range(3):
            print("P(b=" + str(b + 1) + "|c=" + str(c + 1) + ",d=" + str(d + 1) + ")=" + str(p_b_cd[b][c][d]))

# probability distribution of a.
p_a_be = get_p_a_be()
for b in range(3):
    for e in range(2):
        for a in range(2):
            print("P(a=" + str(a + 1) + "|b=" + str(b + 1) + ",e=" + str(e + 1) + ")=" + str(p_a_be[a][b][e]))
