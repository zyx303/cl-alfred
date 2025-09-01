import torch

# 加载 vocab 对象
vocab_dict = torch.load('./data/json_feat_2.1.0/pp.vocab', map_location='cpu')
word_vocab = vocab_dict['word']
# print(word_vocab.word2index('turn'))
print(word_vocab.index2word(34))
# print(word_vocab.word2index('coach.'))

sentence = [[30, 38, 26, 24, 18, 4, 38, 18, 39, 4, 58, 60, 18, 4, 38, 21, 4, 96, 26, 19, 21, 4, 78, 16]]
# 打印句子对应的词
print(" ".join([word_vocab.index2word(idx) for idx in sentence[0]]))


# print("\n--- 方法二：通过低位索引反向查找 ---")
# # 查看索引 0 到 9 对应的词
# for i in range(10):
#     try:
#         word = word_vocab.index2word(i)
#         print(f"Index: {i} -> Token: '{word}'")
#     except IndexError:
#         # 如果索引超出了词汇表范围，就停止
#         print(f"Index {i} is out of bounds. Vocabulary size is likely {i}.")
#         break