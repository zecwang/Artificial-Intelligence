def filtering(evidence_data_add, prior, total_day):
    # you need to implement this method.

    x_prob_rain = []
    # x_prob_sunny[i] = 1 - x_prob_rain[i]

    # transition model
    yesterday_rain = [0.7, 0.3]
    yesterday_sun = [0.3, 0.7]

    # sensor model
    take_umbrella = [0.9, 0.2]
    no_umbrella = [0.1, 0.8]

    data = []
    with open(evidence_data_add, 'r') as f:
        for line in f.readlines():
            data.append(line.strip().split('\t'))

    if total_day >= 1:
        p_transition = [yesterday_rain[0] * prior[0] + yesterday_sun[0] * prior[1],
                        yesterday_rain[1] * prior[0] + yesterday_sun[1] * prior[1]]
        if data[0][1].startswith('n'):
            p_sensor = no_umbrella
        else:
            p_sensor = take_umbrella
        p = [p_sensor[0] * p_transition[0], p_sensor[1] * p_transition[1]]
        x_prob_rain.append(p[0] / sum(p))

    for i in range(1, total_day):
        p_transition = [yesterday_rain[0] * x_prob_rain[i - 1] + yesterday_sun[0] * (1 - x_prob_rain[i - 1]),
                        yesterday_rain[1] * x_prob_rain[i - 1] + yesterday_sun[1] * (1 - x_prob_rain[i - 1])]
        if data[i][1].startswith('n'):
            p_sensor = no_umbrella
        else:
            p_sensor = take_umbrella
        p = [p_sensor[0] * p_transition[0], p_sensor[1] * p_transition[1]]
        x_prob_rain.append(p[0] / sum(p))

    return x_prob_rain


# following lines are main function:
evidence_data_add = "data//assign2_umbrella.txt"
total_day = 100
# the prior distribution on the initial state, P(X0). 50% rainy, and 50% sunny on day 0.
prior = [0.5, 0.5]

if __name__ == '__main__':
    x_prob_rain = filtering(evidence_data_add, prior, total_day)
    for i in range(100):
        print("Day " + str(i + 1) + ": rain " + str(x_prob_rain[i]) + ", sunny " + str(1 - x_prob_rain[i]))
