import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import spearmanr

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items) +"\n"

train_data, test_data = cv.train_test_split(df, test_size=0.25)
train_data = df

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#Use cosine similarity as a metric
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')



############                COLLABORATIVE              ################################



#Function to return ratings
def predict(ratings, similarity, type='user', mode=0):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])#cuz mean user rating should have same format as ratings
        if mode: pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
	else: pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
	mean_user_rating = ratings.mean(axis=0)
        ratings_diff = (ratings - mean_user_rating[np.newaxis])#cuz mean user rating should have same format as ratings
	#print(ratings_diff.shape, similarity.shape)
        if mode: pred = mean_user_rating[np.newaxis] + ratings_diff.dot(similarity) / np.array([np.abs(similarity).sum(axis=0)])
	else: pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=0)])
    return pred

s = time.time()
item_prediction = predict(train_data_matrix, item_similarity, type='item')
e1 = time.time() - s
s = time.time()
user_prediction = predict(train_data_matrix, user_similarity, type='user')
e2 = time.time() - s

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

def sper(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return 1-((6.0*(mean_squared_error(prediction, ground_truth)))/((len(prediction))**2-1))

print '\n\n1a. User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'User-based CF TopK: ' + str(rmse(user_prediction, test_data_matrix[0:20]))
print 'User-based CF Spearman: ' + str(sper(user_prediction, test_data_matrix))
print "Time taken is: " + str(e1)+".\n"
print '1b. Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))
print 'Item-based CF TopK: ' + str(rmse(item_prediction, test_data_matrix[0:20]))
print 'Item-based CF Spearman: ' + str(sper(item_prediction, test_data_matrix))
print "Time taken is: " + str(e2)+".\n"

s = time.time()
item_prediction = predict(train_data_matrix, item_similarity, type='item', mode = 1)
e1 = time.time() - s
user_prediction = predict(train_data_matrix, user_similarity, type='user', mode = 1)
e2 = time.time() - s

print '2a. User-based CF with baseline estimation RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'User-based CF with baseline estimation TopK: ' + str(rmse(user_prediction, test_data_matrix[0:30]))
print 'User-based CF with baseline estimation Spearman: ' + str(sper(user_prediction, test_data_matrix))
print "Time taken is: " + str(e1)+".\n"
print '2b. Item-based CF RMSE with baseline estimation : ' + str(rmse(item_prediction, test_data_matrix))
print 'Item-based CF TopK with baseline estimation : ' + str(rmse(item_prediction, test_data_matrix[0:30]))
print 'Item-based CF Spearman with baseline estimation : ' + str(sper(item_prediction, test_data_matrix[0:30]))
print "Time taken is: " + str(e2)+".\n"



############                SVD              ################################


def svd(m):
	prod = m.T.dot(m)
	a, b = np.linalg.eigh(prod)
	ordering = a.argsort()[::-1]
	b=b[:,ordering]
	a=a[ordering]
	sigma = np.diag(a)
	prod = m.dot(b)
	for i in xrange(min(sigma.shape)): 
		sigma[i][i] = sqrt(abs(sigma[i][i]))
	for i in xrange(min(len(prod),len(a))): prod[:,i]/=sigma[i][i]
	#print(prod)	
	return prod, sigma, b.T

def svd_90(m):
	prod = m.T.dot(m)
	a, b = np.linalg.eigh(prod)
	sum0 = 0
	sum1=0
	ind = 0
	for i in range(len(a)):
		sum0 += a[i]*a[i]
	sum0 = 0.9*sum0
	for i in range(len(a)):
		sum1 += a[i]*a[i]
		if(sum1 >= sum0):
			ind = i
			break
	for i in range(len(a)):
		if(i>=ind):
			a[i]=0	
	ordering = a.argsort()[::-1]
	b=b[:,ordering]
	a=a[ordering]
	sigma = np.diag(a)
	prod = m.dot(b)
	for i in xrange(min(sigma.shape)): 
		sigma[i][i] = sqrt(abs(sigma[i][i]))
	for i in xrange(min(len(prod),len(a))): prod[:,i]/=sigma[i][i]
	#print(prod)	
	return prod, sigma, b.T

sy = time.time()
u, s, vt = svd(train_data_matrix)
pred = (u.dot(s.dot(vt)))
a1 = time.time() - sy

sy = time.time()
u1, s1, vt1 = svd_90(train_data_matrix)
pred1 = (u1.dot(s1.dot(vt1)))
a2 = time.time() - sy

print '3. SVD RMSE: ' + str(rmse(pred, test_data_matrix))
print 'SVD TopK: ' + str(rmse(pred, test_data_matrix[0:20]))
print 'SVD Spearman: ' + str(sper(pred, test_data_matrix))
print "Time taken is: " + str(a1)+".\n"
print "4. SVD_90 RMSE: " + str(rmse(pred1, test_data_matrix))
print "SVD_90 TopK: " + str(rmse(pred1, test_data_matrix[0:20]))
print "SVD_90 Spearman: " + str(sper(pred1, test_data_matrix))
print "Time taken is: " + str(e2)+".\n"



############                CUR              ################################



def cur(m, k):
	pc = []
	pr = []
	for i in range(len(m)):
		r = m[i]
		pr.append(1.0*np.sum(m[i]**2)/np.sum(m**2))
	for i in range(len(m.T)):
		c = m[:,i]
		pc.append(1.0*np.sum(m[:,i]**2)/np.sum(m**2))
	#print pc
	#print pr
	sel_rows = np.random.choice(len(m),k,p=pr)
	#print sorted(sel_rows)	
	sel_cols = np.random.choice(len(m.T),k,p=pc)
	uni_r = set(sel_rows)
	uni_c = set(sel_cols)
	cnr = []
	cnc = []
	for ur in uni_r:
		cnr.append(sel_rows.tolist().count(ur))
	for uc in uni_c:
		cnc.append(sel_cols.tolist().count(uc))
	#print cnc
	#print cnr
	R = [(1.0*m[i]*sqrt(cnr[list(uni_r).index(i)]))/sqrt(k*pr[i]) for i in uni_r]
	#print R
	#TODO C
	C = [(1.0*m[:,i]*sqrt(cnc[list(uni_c).index(i)]))/sqrt(k*pc[i]) for i in uni_c]
	W = m[:,list(uni_c)]
	W = W[list(uni_r),:]
	ua,ea,vta = svd(W)
	ea[ea!=0] = 1/ea[ea!=0]
	ua = (vta.T.dot(ea.dot(ua.T)))
	ea = ea.T
	return C, ua, R

a = time.time()
u, s, vt = cur(train_data_matrix,10)
s = np.array(s)
u = np.array(u)
pred = (u.T.dot(s.dot(vt)))
e = time.time() - a
print '5. CUR1 RMSE: ' + str(rmse(pred, test_data_matrix))
print 'CUR1 TopK: ' + str(rmse(pred, test_data_matrix[0:20]))
print 'CUR1 Spearman: ' + str(sper(pred, test_data_matrix[0:20]))
print "Time taken is: " + str(e)+".\n"



def cur_2(m, k):
	pc = []
	pr = []
	for i in range(len(m)):
		r = m[i]
		pr.append(1.0*np.sum(m[i]**2)/np.sum(m**2))
	for i in range(len(m.T)):
		c = m[:,i]
		pc.append(1.0*np.sum(m[:,i]**2)/np.sum(m**2))
	#print pc
	#print pr
	sel_rows = np.random.choice(len(m),k,p=pr,replace = False)
	#print sorted(sel_rows)	
	sel_cols = np.random.choice(len(m.T),k,p=pc, replace = False)
	uni_r = set(sel_rows)
	uni_c = set(sel_cols)
	cnr = []
	cnc = []
	for ur in uni_r:
		cnr.append(sel_rows.tolist().count(ur))
	for uc in uni_c:
		cnc.append(sel_cols.tolist().count(uc))
	#print cnc
	#print cnr
	R = [(1.0*m[i]*sqrt(cnr[list(uni_r).index(i)]))/sqrt(k*pr[i]) for i in uni_r]
	#print R
	#TODO C
	C = [(1.0*m[:,i]*sqrt(cnc[list(uni_c).index(i)]))/sqrt(k*pc[i]) for i in uni_c]
	W = m[:,list(uni_c)]
	W = W[list(uni_r),:]
	ua,ea,vta = svd(W)
	ea[ea!=0] = 1/ea[ea!=0]
	ua = (vta.T.dot(ea.dot(ua.T)))
	ea = ea.T
	return C, ua, R

a = time.time()
u, s, vt = cur_2(train_data_matrix,10)
s = np.array(s)
u = np.array(u)
pred = (u.T.dot(s.dot(vt)))
e = time.time() - a
print '6. CUR2 RMSE: ' + str(rmse(pred, test_data_matrix))
print 'CUR2 TopK: ' + str(rmse(pred, test_data_matrix[0:20]))
print 'CUR2 Spearman: ' + str(sper(pred, test_data_matrix))
print "Time taken is: " + str(e)+".\n"
