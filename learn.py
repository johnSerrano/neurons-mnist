from pyfann import libfann

learning_rate = 0.4
num_input = 784
num_hidden_1 = 100
num_output = 1

desired_error = 0.0001
max_iterations = 100
iterations_between_reports = 1


def train_digit(i):
    nn = libfann.neural_net()
    nn.create_standard_array((num_input, num_hidden_1, num_output))
    nn.set_learning_rate(learning_rate)
    nn.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)

    nn.train_on_file("digit-" + str(i) + ".data", max_iterations, iterations_between_reports, desired_error)

    nn.save("digit-" + str(i) + ".net")
    print("NET " + str(i) + " DONE")

for i in range(10):
    train_digit(i)
