from pyfann import libfann

nn= libfann.neural_net()
nn.create_from_file("digits.net")

total_correct = 0

with open("t10k-images-idx3-ubyte", "rb") as f:
	with open("t10k-labels-idx1-ubyte", "rb") as g:
		f.read(16)
		g.read(8)
		for i in range(10000):
			pixel_values = []
			for i in range(28**2):
				pixel_values.append((ord(f.read(1)[0]) / 128.0) - 1)
			real_answer = (ord(g.read(1)[0]))
			result = nn.run(pixel_values)
			if result.index(max(result)) == real_answer:
				total_correct += 1

print total_correct/10000.0
