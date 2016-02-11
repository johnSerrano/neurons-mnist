import sys

with open("train-images-idx3-ubyte", "rb") as f:
	with open("train-labels-idx1-ubyte", "rb") as g:
		f.read(16)
		g.read(8)
		for j in range(60000):
			
			for i in range(28**2):
				pixel_value = str((f.read(1)[0] / 128.0) - 1)
				sys.stdout.write(pixel_value + ' ')
			print() 
			print({
				0: "1 -1 -1 -1 -1 -1 -1 -1 -1 -1",
				1: "-1 1 -1 -1 -1 -1 -1 -1 -1 -1",
				2: "-1 -1 1 -1 -1 -1 -1 -1 -1 -1",
				3: "-1 -1 -1 1 -1 -1 -1 -1 -1 -1",
				4: "-1 -1 -1 -1 1 -1 -1 -1 -1 -1",
				5: "-1 -1 -1 -1 -1 1 -1 -1 -1 -1",
				6: "-1 -1 -1 -1 -1 -1 1 -1 -1 -1",
				7: "-1 -1 -1 -1 -1 -1 -1 1 -1 -1",
				8: "-1 -1 -1 -1 -1 -1 -1 -1 1 -1",
				9: "-1 -1 -1 -1 -1 -1 -1 -1 -1 1"
			}[(g.read(1)[0])])
			print()
