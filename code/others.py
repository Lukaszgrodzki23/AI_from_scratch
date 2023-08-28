import math
from NN import Network, Activation_layer, Dense_layer

def check_performance(x_train, y_train, epochs, activation_funcs, loss_funcs, learn_rates, numbers_of_nodes, input_size, output_size):
    min_error = math.inf
    summary = []
    for loss_func in loss_funcs:
        for number_of_nodes in numbers_of_nodes:
            for learn_rate in learn_rates:
                net = Network()
                net.add_layer(Dense_layer(input_size, number_of_nodes))
                net.add_layer(Activation_layer(activation_funcs[0][0], activation_funcs[0][1]))
                for i in range(1, len(activation_funcs) - 1):
                    net.add_layer(Dense_layer(number_of_nodes, number_of_nodes))
                    net.add_layer(Activation_layer(activation_funcs[i][0], activation_funcs[i][1]))
                net.add_layer(Dense_layer(number_of_nodes, output_size))
                net.add_layer(Activation_layer(activation_funcs[-1][0], activation_funcs[-1][1]))
                net.choose_loss_func(loss_func[0], loss_func[1])
                error = net.fit(x_train, y_train, epochs, learn_rate, False)
                summary.append([error, learn_rate, number_of_nodes, loss_func[0]])
                if min_error > error:
                    min_error = error
                    best_net = net
                    params = [learn_rate, number_of_nodes, loss_func[0]]
    return summary, min_error, params
