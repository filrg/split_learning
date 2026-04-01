
def partition(exe_time_layer_1, net_layer_1, exe_time_layer_2, net_layer_2, size_data):
    t_max = 0
    result = 0
    for cut_point in range(len(exe_time_layer_1)):
        speed_layer_1 = 0
        speed_layer_2 = 0
        size_d = size_data[cut_point]
        for exe, comm in zip(exe_time_layer_1, net_layer_1):
            speed_layer_1 += (1 / (exe[cut_point] + (2 * (size_d / comm))))
        for exe, comm in zip(exe_time_layer_2, net_layer_2):
            speed_layer_2 += (1 / (exe[cut_point] + (2 * (size_d / comm))))
        speed = min(speed_layer_1, speed_layer_2)
        if speed > t_max:
            result = cut_point
            t_max = speed
    result = result + 1

    return [result]