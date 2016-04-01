import sys

def main():
	for i in range (10):
		create_net(i)


def create_net(i):
	with open("train-images-idx3-ubyte", "rb") as f:
		with open("train-labels-idx1-ubyte", "rb") as g:
			filename = "digit-" + str(i) + ".data"
			with open(filename, 'w') as out:
				towrite = '60000 784 1\n'
				f.read(16)
				g.read(8)
				for j in range(60000):
					if j%1000 == 0:
						print(j)
					for k in range(28**2):
						pixel_value = str((f.read(1)[0] / 128.0) - 1)
						towrite += pixel_value + ' '
					towrite += "\n"
					tmp = g.read(1)[0]
					if tmp == i:
						towrite += "1\n"
					else:
						towrite += "-1\n"
				out.write(towrite)
				print("Wrote " + filename)

if __name__ == "__main__":
	main()
