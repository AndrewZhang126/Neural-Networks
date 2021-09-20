# The following code was adapted from Week 2 Programming Assignment 1 in the Sequence Models course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/nlp-sequence-models/home/week/2



import numpy as np
#GloVe vectors
words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')



"""
This function computes the cosine similarity between two word vectors
"""
def cosine_similarity(u, v):

    # u and v are word vectors
    
    if np.all(u == v):
        return 1
    
    # compute the dot product between u and v 
    dot = np.dot(u, v) 
    
    # compute the L2 norm of u and v
    norm_u = np.sqrt(np.sum(pow(u, 2)))
    norm_v = np.sqrt(np.sum(pow(v, 2)))
    
    # avoid division by 0
    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0
    
    # compute the cosine similarity
    cosine_similarity = dot / (norm_u * norm_v)
    
    return cosine_similarity



"""
This function performs the word analogy task "a is to b as c is to ___"
"""
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):

    # word_a,  word_b, and word_c are words (strings)
    # word_to_vec_map is a dictionary that maps words to their corresponding vectors. 
    
    # convert words to lowercase
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    # get the word embeddings 
    e_a , e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              
    best_word = None                   
    
    # loop over the whole word vector set
    for w in words:   
    
        if w == word_c:
            continue
        
        # compute the cosine similarity between the vectors
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)
        
        # update the maximum cosine similarity value when applicable
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word



# test the code
triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad, word_to_vec_map)))



# display examples of gender bias in word embeddings, 
# where negative values correspond to males and postive values correspond to females
print('Words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))



"""
This function removes bias of words to ensure gender neutral words 
"""
def neutralize(word, g, word_to_vec_map):

    # select word vector representation
    e = word_to_vec_map[word]
    
    # compute bias component
    e_biascomponent = (np.dot(e, g) / np.sum(g * g)) * g
 
    # neutralize 
    e_debiased = e - e_biascomponent
    
    return e_debiased



"""
This functiond debiases gender specific words using the equalization algorithm
"""
def equalize(pair, bias_axis, word_to_vec_map):

    # pair is the gender specific words to debias
    # bias_axis is the bias axis (gender)
    
    # select word vector representation
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    # compute the mean of the word vectors
    mu = (e_w1 + e_w2) / 2

    # compute the projection over the bias axis and the orthogonal axis
    mu_B = (np.dot(mu, bias_axis) / np.sum(bias_axis * bias_axis)) * bias_axis
    mu_orth = mu - mu_B

    # compute and adjust the bias 
    e_w1B = (np.dot(e_w1, bias_axis) / np.sum(bias_axis * bias_axis)) * bias_axis
    e_w2B = (np.dot(e_w2, bias_axis) / np.sum(bias_axis * bias_axis)) * bias_axis
    corrected_e_w1B = np.sqrt(np.abs(1 - np.sum(mu_orth * mu_orth))) * ((e_w1B - mu_B) / np.linalg.norm((e_w1 - mu_orth) - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.sum(mu_orth * mu_orth))) * ((e_w2B - mu_B) / np.linalg.norm((e_w2 - mu_orth) - mu_B))

    # debias using equalization
    e2 = corrected_e_w2B + mu_orth                                                     
    
    return e1, e2


# print results of equalization algorithms
print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))



# References
# [1] https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf
# [2] https://nlp.stanford.edu/projects/glove/
