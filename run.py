import numpy as np

test_1 = np.concatenate((np.random.normal(size=1,loc=1,scale=2),np.random.normal(size=1,loc=8,scale=2),np.random.normal(size=1,loc=15,scale=2),np.random.normal(size=1,loc=25,scale=2)))
test_2 = np.concatenate((np.random.normal(size=1,loc=15,scale=2),np.random.normal(size=1,loc=1,scale=2),np.random.normal(size=1,loc=20,scale=2),np.random.normal(size=1,loc=0,scale=2)))
X_test = np.stack([test_1,test_2],1)
#検証データ
#[3, 0, 1, 2]

#import matplotlib.pyplot as plt
#import itertools 
import kmeans as k
#import sckmeans as k 

kmeans = k.kmeans(K=4,iter=10,random_state=123)

data = kmeans.generate()
print(data)
kmeans.fit(data)

#print(kmeans.predict(X_test))

#kmeans.class_plot()


#print(kmeans.his_id)
#print(kmeans.his_id[4])
#print(kmeans.his_id[9]==kmeans.his_id[0])

#print(kmeans.his_centers)
#print(kmeans.his_centers[9]==kmeans.his_centers[1])

kmeans.plot_animation()