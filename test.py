from pyfann import libfann

#slow

total_correct = 0
nets = []
for i in range(10):
	nn = libfann.neural_net()
	nn.create_from_file("digit-" + str(i) + ".net")
	nets.append(nn)

fails = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

with open("t10k-images-idx3-ubyte", "rb") as f:
	with open("t10k-labels-idx1-ubyte", "rb") as g:
		f.read(16)
		g.read(8)
		for i in range(10000):

			pixel_values = []
			for k in range(28**2):
				pixel_values.append((ord(f.read(1)[0]) / 128.0) - 1)
			real_answer = (ord(g.read(1)[0]))

			results = []
			for j in range(10):
				results.append(nets[j].run(pixel_values)[0])

			if i % 100 == 0:
				print(i)


			if results.index(max(results)) == real_answer:
				total_correct += 1
			else:
				fails[real_answer] += 1
print "Fails:"
print fails
print total_correct/10000.0
