from sklearn.feature_extraction.text import CountVectorizer
import os

max_features = 5000

#每个邮件为一个字符串，过滤回车和换行
def load_one_file(filename):
    x = ""
    with open(filename,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            x += line
    return x

#遍历指定文件夹下所有文件，加载数据
def load_files_from_dir(rootdir):
    x=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v=load_one_file(path)
            x.append(v)
    return x

#加载文件夹
def load_all_files():
    ham=[]
    spam=[]
    for i in range(1,7):
        path = "Enron/enron%d/ham/" % i
        print("Load %s" % path)
        ham += load_files_from_dir(path)
        path = "Enron/enron%d/spam/" % i
        print("Load %s" % path)
        spam += (load_files_from_dir(path))
    return ham,spam


#词袋模型，文字样本向量化，ham:0，spam:1
def get_features():
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    return x,y
