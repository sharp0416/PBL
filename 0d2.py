import numpy as np
import sys

class Sigmoid:              #シグモイド関数のクラス
    def __init__(self):
        self.params,self.grads=[],[]
        self.out=None

    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out
    
    def backward(self,dout):
        dx=dout*(1.0-self.out)*self.out
        return dx

class Affine:                   #全結合層に関するクラス
    def __init__(self,W,b):
        self.params=[W,b]       #インスタンスにparamsという配列を作り、ここに引数W（重み）、b（バイアス）を格納する。
    
    def forward(self,x):
        W,b=self.params
        out=np.dot(x,W)+b       #重みと入力の内積を計算し、バイアスを足したものを出力とする。
        return out

class MatMul:
    def __init__(self,W):
        self.params=

def preprocess(text):
    text=text.lower()
    text=text.replace('.',' .')
    words=text.split(' ')

    word_to_id={}
    id_to_word={}
    for word in words:
        if word not in word_to_id:
            new_id=len(word_to_id)
            word_to_id[word]=new_id
            id_to_word[new_id]=word
    
    corpus=np.array([word_to_id[w] for w in words])

    return corpus,word_to_id,id_to_word

def creat_contexts_target(corpus,window_size=1):
    target=corpus[window_size:-window_size]
    contexts=[]
    for idx in range(window_size,len(corpus)-window_size):
        cs=[]
        for t in range(-window_size,window_size+1):
            if t==0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    return np.array(contexts),np.array(target)

class RNN:
    def __init__(self,Wx,Wh,b):     #初期化関数
        self.params=[Wx,Wh,b]
        self.grads=[np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.cache=None             #逆伝搬時に使う中間データ（初期化時はなし）

    def forward(self,x,h_prev):     #フォア―ド計算
        Wx,Wh,b=self.params
        t=np.dot(h_prev,Wh)+np.dot(x,Wx)+b
        h_next=np.tanh(t)

        self.cache=(x,h_prev,h_next)
        return h_next
    
    def backward(self,dh_next):
        Wx,Wh,b=self.params
        x,h_prev,h_next=self.cache

        dt=dh_next*(1-h_next**2)
        db=np.sum(dt,axis=0)        #axis=0は行列において各列の要素に関する処理。sumの場合は各列の要素を足しこみベクトルとする。
        dWh=np.dot(h_prev.T,dt)
        dh_prev=np.dot(dt,Wh.T)
        dWx=np.dot(x.T,dt)
        dx=np.dot(dt,Wx.T)

        self.grads[0][...]=dWx      #[...]は行列において0行目の要素のすべてを表す。
        self.grads[1][...]=dWh
        self.grads[2][...]=db

        return dx,dh_prev

class TimeRNN:
    def __init__(self,Wx,Wh,b,stateful=False):      #pythonでは慣習的にコンストラクタを__init__で書く。
        self.params=[Wx,Wh,b]
        self.grads=[np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.layers=None

        self.h,self.dh=None,None
        self.stateful=stateful

    def set_state(self,h):
        self.h=h

    def reset_state(self):
        self.h=None

