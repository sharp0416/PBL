import numpy as np
import sys
import matplotlib.pyplot as plt

class Sigmoid:              #シグモイド関数のクラス
    def __init__(self):     #コンストラクタ
        self.params,self.grads=[],[]        #パラメータと重みをからの配列で初期化
        self.out=None                       #出力はなし（コンストラクタだから？）

    def forward(self,x):            #フォワード計算のメソッド
        out=1/(1+np.exp(-x))        #outは出力（シグモイドの結果だから0から1の値）
        self.out=out            #「self.」は「そのインスタンスの」という意味。今回はそのインスタンスの出力にoutを代入という意味。
        return out
    
    def backward(self,dout):                #バックワード計算のメソッド
        dx=dout*(1.0-self.out)*self.out     #後述
        return dx

class Affine:                   #全結合層に関するクラス
    def __init__(self,W,b):     #コンストラクタ
        self.params=[W,b]       #インスタンスにparamsという配列を作り、ここに引数W（重み、行列の形）、b（バイアス、ベクトルの形）を格納する。
        self.grads=[np.zeros_like(W),np.zeros_like(b)]      #勾配をそれぞれW,bの形を保ったゼロ行列、ゼロベクトルで初期化
        self.x=None       #とりあえず無しにしておく、xは入力 
    
    def forward(self,x):        #フォワード
        W,b=self.params         #インスタンスの変数をローカル変数に代入
        out=np.dot(x,W)+b       #重みと入力の内積を計算し、バイアスを足したものを出力とする。
        self.x=x            #引数（入力）をインスタンスに代入する
        return out          #積を返す

    def backward(self,dout):    #バックワードのメソッド
        W,b=self.params         #後述
        dx=np.dot(dout,W.T)
        dW=np.dot(self.x.T,dout)
        db=np.sum(dout,axis=0)

        self.grads[0][...]=dW
        self.grads[1][...]=db
        return dx

class MutMul:                   #行列積のメソッド？
    def __init__(self,W):       #コンストラクタ
        self.params=[W]         #
        self.grads=[np.zeros_like(W)]
        self.x=None
    
    def forward(self,x):
        W,=self.params
        out=np.dot(x,W)         #二つの入力Wとxの行列積計算「dot」は内積もしくは行列積の計算
        self.x=x
        return out

    def backward(self,dout):
        W,=self.params
        dx=np.dot(dout,W.T)         #dxはxでの偏微分（「W.T」はWの転置を表す）
        dW=np.dot(self.x.T,dout)    #dWはWでの偏微分
        self.grads[0][...]=dW       #←はgrads[0]=dWと同じ意味だが3点リーダーの方はディープコピー、後者はshallow copy
        return dx                   #二つのコピーの詳細は以下

        #---------------------------------------------------------------------------------------------------------------
        #   a=np.array([1,2,3]),b=np.array([4,5,6])の時                                                                 
        #                                                                                                a,b            
        #                                                                                                ↓  
        #                                                      ____(shallow　copy)___   ・・・ 1,2,3・・・4,5,6・・・
        #                   a          b                      |
        #                   ↓          ↓                      |
        #   メモリ：   ・・・1,2,3・・・4,5,6・・・       -------                               a          b
        #                                                     |                               ↓          ↓
        #                                                     |____(deep copy)_______   ・・・ 4,5,6・・・4,5,6・・・
        # 
        #--------------------------------------------------------------------------------------------------------------- 

class SGD:                      #更新に関するクラス
    def __init__(self,lr=0.01):     #コンストラクタ（lrは学習率）
        self.lr=lr                  #引数をインスタンスに渡す
    
    def update(self,params,grads):      #更新に関するメソッド
        for i in range(len(params)):    #すべてのパラメータに関してforループで更新を行う
            params[i]-=self.lr*grads[i]     #w_(new)=w_(old)-η×（∂E/∂w)

class SoftmaxWithLoss:              #ソフトマックスと交差エントロピー誤差を組み合わせたクラス
    def __init__(self):     #コンストラクタ
        self.params,self.grads=[],[]    #パラメータと勾配の配列を用意
        self.y=None             #softmaxの出力（とりあえずNone）
        self.t=None             #教師ラベル（とりあえずNone）

    def forward(self,x,t):      #フォワード計算
        self.t=t                #インスタンスにローカル変数t（教師データ）を渡す
        self.y=softmax(x)       #インスタンスのyにソフトマックスの結果を格納

        if self.t.size==self.y.size:
            self.t=self.t.argmax(axis=1)

        loss=cross_entropy_error(self.y,self.t)
        return loss

    def backward(self,dout=1):
        batch_size=self.t.shape[0]

        dx=self.y.copy()
        dx[np.arange(batch_size),self.t]-=1
        dx*=dout
        dx=dx/batch_size

        return dx
    
def softmax(x):                 #ソフトマックスを行う関数
    #---------------------------------------------------------------------------------------------------------------
    #ソフトマックスについて
    #   ソフトマックスは出力関数の一種であり、分類問題など確率分布的な出力が期待されるとき、出力値の合計を1とする
    #   ものである。ソフトマックスの出力は以下で与えられる。
    #
    #                    exp(a_k)
    #           y_k = ----------------      （Σは入力データ全範囲に対して）
    #                    Σexp(a_i)
    #
    #---------------------------------------------------------------------------------------------------------------
    if x.ndim==2:               #引数が2次元（行列形式）の時
        x=x-x.max(axis=1,keepdims=True)
        x=np.exp(x)
        x/=x.sum(axis=1,keepdims=True)
    elif x.ndim==1:             #引数が1次元（ベクトル形式）の時
        x=x-np.max(x)
        x=np.exp(x)/np.sum(np.exp(x))

    return x

def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

        if t.size==y.size:
            t=t.argmax(axis=1)
        
    batch_size=y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size






#スパイラルデータセット（開始）
sys.path.append('../')          #親ディレクトリをインポート
from dataset import spiral      #datasetデータのspiral.pyをインポート

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        I,H,O=input_size,hidden_size,output_size

        #重みとバイアスの初期化
        W1=0.01*np.random.rand(I,H)
        b1=np.zeros(H)
        W2=0.01*np.random.rand(H,O)
        b2=np.zeros(O)

        #レイヤの生成
        self.layers=[
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]
        
        self.loss_layer=SoftmaxWithLoss()

        self.params,self.grads=[],[]
        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads

    def predict(self,x):
        for layer in self.layers:
            x=layer.forward(x)
        return x
    
    def forward(self,x,t):
        score=self.predict(x)
        loss=self.loss_layer.forward(score,t)
        return loss

    def backward(self,dout=1):
        dout=self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout=layer.backward(dout)
        return dout
    
max_epoch=300
batch_size=30
hidden_size=10
learning_rate=1.0

x,t=spiral.load_data()
model=TwoLayerNet(input_size=2,hidden_size=hidden_size,output_size=3)
optimizer=SGD(lr=learning_rate)

data_size=len(x)
max_iters=data_size//batch_size
total_loss=0
loss_count=0
loss_list=[]

for epoch in range(max_epoch):
    idx=np.random.permutation(data_size)
    x=x[idx]
    t=t[idx]
    for iters in range(max_iters):
        batch_x=x[iters*batch_size:(iters+1)*batch_size]
        batch_t=t[iters*batch_size:(iters+1)*batch_size]

        loss=model.forward(batch_x,batch_t)
        model.backward()
        optimizer.update(model.params,model.grads)

        total_loss+=loss
        loss_count+=1

        if (iters+1)%10==0:
            avg_loss=total_loss/loss_count
            print('| epoch %d | iter %d / %d | loss %2.f'%(epoch+1,iters+1,max_iters,avg_loss))
            loss_list.append(avg_loss)
            total_loss,loss_count=0,0

# 境界領域のプロット
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# データ点のプロット
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()




#スパイラルデータセット（終了）

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


