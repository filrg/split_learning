
def partition(model_name, exe_time_layer_1, net_layer_1, exe_time_layer_2, net_layer_2, size_data):
    if model_name == 'Bert':
        t_max = 0
        result = 0
        for cut_point in range(len(exe_time_layer_1)):
            speed_layer_1 = 0
            speed_layer_2 = 0
            for exe, comm in zip(exe_time_layer_1, net_layer_1):
                speed_layer_1 += (1 / (exe[cut_point] + (2 * (size_data / comm))))
            for exe, comm in zip(exe_time_layer_2, net_layer_2):
                speed_layer_2 += (1 / (exe[cut_point] + (2 * (size_data / comm))))
            speed = min(speed_layer_1, speed_layer_2)
            if speed > t_max:
                result = cut_point
                t_max = speed
        result = result + 1
    else:
        t_max = 0
        result = 0
        for cut_point in range(1, len(size_data) - 1):
            speed_layer_1 = 0
            speed_layer_2 = 0
            for exe, comm in zip(exe_time_layer_1, net_layer_1):
                speed_layer_1 += (1 / (sum(exe[:cut_point]) + (2 * (size_data[cut_point - 1] / comm))))
            for exe, comm in zip(exe_time_layer_2, net_layer_2):
                speed_layer_2 += (1 / (sum(exe[cut_point:]) + (2 * (size_data[cut_point - 1] / comm))))
            speed = min(speed_layer_1, speed_layer_2)
            if speed > t_max:
                result = cut_point
                t_max = speed

    return [result]
