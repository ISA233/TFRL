import pickle
from tools import vector


def load_data(path='train.pkl'):
	data = pickle.load(open(path, 'rb'))
	for _data in data:
		v = _data[3] * 2 - 1
		a = _data[2]
		if a == -1:
			a = 64
		_data[2] = v
		_data[3] = vector(a)
	# print(_data)
	return data


data = load_data()
print(data[0])
with open('train_new.pkl', 'wb') as f:
	pickle.dump(data, f)
