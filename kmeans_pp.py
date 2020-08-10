import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

#Generate data
#https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
#https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html


class kmeans:
    def __init__(self, K, iter=10, random_state=0):
        self.iter = iter
        self.K = K
        self.random_state = np.random.seed(random_state)

        self.his_id = []
        self.his_centers = []

    def generate(self):
        '''
        Generate sample data.
        Four mixed Gaussian distributions are generated.
        shape is (80,2)
        '''
        x1 = np.concatenate((np.random.normal(size=20,loc=1,scale=2),np.random.normal(size=20,loc=8,scale=2),np.random.normal(size=20,loc=15,scale=2),np.random.normal(size=20,loc=25,scale=2)))
        x2 = np.concatenate((np.random.normal(size=20,loc=15,scale=2),np.random.normal(size=20,loc=1,scale=2),np.random.normal(size=20,loc=20,scale=2),np.random.normal(size=20,loc=0,scale=2)))
        
        x1 = np.concatenate((np.random.normal(size=20,loc=1,scale=2),np.random.normal(size=20,loc=8,scale=2),np.random.normal(size=20,loc=15,scale=2),np.random.normal(size=20,loc=25,scale=2)))
        x2 = np.concatenate((np.random.normal(size=20,loc=15,scale=3),np.random.normal(size=20,loc=1,scale=3),np.random.normal(size=20,loc=20,scale=3),np.random.normal(size=20,loc=0,scale=3)))
        
        X = np.stack([x1,x2],1)
        #shuffle
        np.random.shuffle(X)
        return X
    
    def fit(self,X):
        '''
        After initializing the id and centroid
        The id updates and centroid updates are repeated.
        '''        
        #データをクラス全体に反映
        self.X = X

        #データ数分のzeroラベルをつくる
        #これを書き換えていく（新しいクラス）
        self.id = np.zeros(X.shape[0])

        '''
        #セントロイドの初期化。
        #適当な位置での初期値だと１つもデータを含まない点で初期化される可能性があり、.meanでinvalid_devideエラーを吐く可能性がある。
        #そのために、ランダムに取ってきたデータをセントロイド
        feature_indexes = np.arange(self.X.shape[0])
        #np.random.shuffle(feature_indexes) 
        #initial_centroid_indexes = feature_indexes[:self.K]
        initial_centroid_indexes = np.random.choice(feature_indexes, self.K, replace=False)
        self.centers = self.X[initial_centroid_indexes]
        '''

        #### K-means++のcenterの初期化方法
        distance = np.zeros(self.X.shape[0]*self.K).reshape(self.X.shape[0],self.K)
        self.centers = np.zeros(2*self.K).reshape(self.K,-1)

        for i in range(self.K):
            #1個目のセントロイドのみ一律に選ばれて欲しい
            if i == 0:
                p = np.repeat(1/self.X.shape[0],self.X.shape[0])
            else:
                p = np.sum(distance,axis=1)/np.sum(distance)
 
            #一個目の中心の設定
            #初回は同じ確率で選ばれて欲しい。
            self.centers[i, :] = self.X[np.random.choice(np.arange(self.X.shape[0]), 1, p=p),:]
            distance[:, i] = np.sum((self.X - self.centers[i, :])**2,axis=1)
            
        
        #更新スタート
        for _ in range(self.iter):
            #一番近い点を探して、クラスを変更する
            for i in range(self.X.shape[0]):
                self.id[i] = np.argmin(np.sum((self.X[i,:] - self.centers)**2,axis=1))
            #idの更新履歴を残す。
            #クラス番号なので、int型で欲しい＝＞tolist()メソッドを使うとfloat型になってしまう。
            int_id_list = [int(k) for k in self.id.tolist()]
            self.his_id.append(int_id_list)

            #クラスが決まったので、そのクラスの中の平均の場所にセントロイドを移動させる。
            for k in range(self.K):
                #idがTrueの行のみ取ってくる
                self.centers[k,:] = self.X[self.id==k,:].mean(axis=0)
                
            c = self.centers.tolist()
            self.his_centers.append(c)
        #最終的にはnumpyで扱いたいので、更新終了後に型を変換
        self.his_id = np.array(self.his_id)
        self.his_centers = np.array(self.his_centers)
        
    def predict(self,X):
        '''
        Returns a list of classification results for the test data.
        '''        
        la_list = []
        for i in range(X.shape[0]):
            la_list.append(np.argmin(np.sum((X[i,:] - self.centers)**2,axis=1)))
        return la_list

    def class_plot(self):
        '''
        Display the result of classification with matplotlib.
        Each class datas are colored.
        '''
        fig = plt.figure()
        ax = fig.add_subplot()
        random = np.random.rand(1)

        for j in range(self.K):
            random = np.random.rand(3)
            #R,G,Bを乱数で生成
            rgb_list = [round(random[0],1), round(random[1],1), round(random[2],1)]
            col = [rgb_list for i in range(self.X[self.id==j,0].shape[0])]

            ax.scatter(self.X[self.id==j,0],self.X[self.id==j,1],c=col,s=10,alpha=0.5)
        #重心のプロット
        ax.scatter(self.centers[:, 0],self.centers[:, 1],c=np.random.rand(4), s=30,alpha=1)
        
        #クラス番号のアノテーション
        for i,x,y in zip(list(range(self.K)),self.centers[:,0].tolist(),self.centers[:,1].tolist()):
            ax.annotate(i,xy=(x, y))

        plt.show()
        plt.savefig('fig-kmeans.png')


    def plot_animation(self):
        '''
        吾輩はまだ動かぬ.        
        '''
        #fit関数の１番上に必要
        fig = plt.figure()
        #ax = fig.add_subplot()

        img1 = []
        img2 = []
        center_col = np.random.rand(4)
        for i in range(self.iter):
            img1_ = []
            
            for j in range(self.K):
                random = np.random.rand(3)
                #R,G,Bを乱数で生成
                rgb_list = [round(random[0],1), round(random[1],1), round(random[2],1)]
                col = [rgb_list for i in range(self.X[self.his_id[i]==j,0].shape[0])]

                #img1_s = ax.scatter(self.X[self.his_id[i]==j,0],self.X[self.his_id[i]==j,1],c=col,s=10,alpha=0.5)
                img1_s = plt.scatter(self.X[self.his_id[i]==j,0],self.X[self.his_id[i]==j,1],c=col,s=10,alpha=0.5)
                
                img1_.append(img1_s)
            img1.append([img1_])
            
            #重心のプロット
            #img2_ = ax.scatter(self.his_centers[i][:, 0],self.his_centers[i][:, 1],c=np.random.rand(4), s=30,alpha=1)
            img2_ = plt.scatter(self.his_centers[i][:, 0],self.his_centers[i][:, 1],c=center_col, s=30,alpha=1)
            print(i)
            img2.append([img2_])

        ims = []
        #ims.append(img1)
        ims.append(img2)
        ani = animation.ArtistAnimation(fig, img2, interval=100)
        ani.save('kmeans.gif', writer='pillow')
        plt.show()
