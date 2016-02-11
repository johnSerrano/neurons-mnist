from pyfann import libfann

learning_rate = 0.7
num_input = 784
num_hidden_1 = 15
num_output = 10

desired_error = 0.04
max_iterations = 10000
iterations_between_reports = 10

nn = libfann.neural_net()
nn.create_standard_array((num_input, num_hidden_1, num_output))
nn.set_learning_rate(learning_rate)
nn.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)

nn.train_on_file("mnist_digits.data", max_iterations, iterations_between_reports, desired_error)

nn.save("digits.net")
