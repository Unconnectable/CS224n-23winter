from gzip import FCOMMENT


def plot_embeddings(M_reduced, word2ind, words):
    ### SOLUTION BEGIN
    '''
        M_reduced : 二维词嵌入矩阵 (n_words, 2)
        word2ind : 单词到矩阵索引的映射字典  
        words : 需要可视化的单词列表
    '''
    for word in words:
        idx = word2ind[word]
        #序列解包坐标
        x_coords, y_coords = M_reduced[idx]
          # 绘制红色"x"形散点(marker='x'指定形状)
        plt.scatter(x_coords, y_coords, marker="x", color="purple")
        # 在坐标点旁添加单词标签(fontsize控制字号)
        plt.annotate(word, (x_coords, y_coords), fontsize=12)  
        
    plt.show()
    ### SOLUTION END