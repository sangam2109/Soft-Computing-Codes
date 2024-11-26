import numpy as np

# Define fuzzy sets A and B as dictionaries with element: membership degree
A = {1: 0.1, 2: 0.4, 3: 0.6, 4: 0.9}
B = {1: 0.7, 2: 0.5, 3: 0.2, 4: 0.8}

# Union of fuzzy sets
def fuzzy_union(A, B):
    return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

# Intersection of fuzzy sets    
def fuzzy_intersection(A, B):
    return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) & set(B)}

# Complement of fuzzy set
def fuzzy_complement(A):
    return {x: 1 - A[x] for x in A}

# Difference of fuzzy sets
def fuzzy_difference(A, B):
    return {x: min(A.get(x, 0), 1 - B.get(x, 0)) for x in A}

# Cartesian product of two fuzzy sets
def fuzzy_cartesian_product(A, B):
    return {(x, y): min(A[x], B[y]) for x in A for y in B}

# Max-Min Composition of two fuzzy relations
def max_min_composition(R1, R2, Y):
    result = {}
    for x in set(R1.keys()):
        for z in set(R2.keys()):
            result[(x[0], z[1])] = max(min(R1[x], R2[(y, z[1])]) for y in Y)
    return result

# Perform operations on fuzzy sets A and B
union_AB = fuzzy_union(A, B)
intersection_AB = fuzzy_intersection(A, B)
complement_A = fuzzy_complement(A)
difference_AB = fuzzy_difference(A, B)

# Cartesian product of A and B
R = fuzzy_cartesian_product(A, B)

# Assume another fuzzy relation between sets B and C
C = {1: 0.9, 2: 0.4, 3: 0.3, 4: 0.7}
R1 = fuzzy_cartesian_product(A, B)
R2 = fuzzy_cartesian_product(B, C)

# Perform max-min composition
composition_R1_R2 = max_min_composition(R1, R2, B.keys())

# Print results
print("Fuzzy Union A ∪ B:", union_AB)
print("Fuzzy Intersection A ∩ B:", intersection_AB)
print("Fuzzy Complement A^c:", complement_A)
print("Fuzzy Difference A - B:", difference_AB)
print("Fuzzy Relation (Cartesian Product) A × B:", R)
print("Max-Min Composition R1 and R2",composition_R1_R2)
