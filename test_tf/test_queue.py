from MCTS import Node

print(233)

a = Node()
a.edges = [(0, 0, Node(a)), (1, 0, Node(a)), (2, 0, Node(a))]
for i in range(3):
	print(i)
	a.son(i).edges = [(0, 0, Node(a.son(i))), (1, 0, Node(a.son(i)))]

print('root:', a)
print(a.edges)
for i in range(3):
	print(a.son(i), a.son(i).edges)
a.move_root(1)
print(a.father)
print(a.son(0).father, a.son(1).father)

while True:
	pass
