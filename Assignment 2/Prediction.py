from Filtering import filtering


def prediction(evidence_data_add, prior, start_day, end_day):
    # you need to implement this method.

    x_prob_rain = filtering(evidence_data_add, prior, start_day - 1)
    # x_prob_sunny[i] = 1 - x_prob_rain[i]

    # transition model
    yesterday_rain = [0.7, 0.3]
    yesterday_sun = [0.3, 0.7]

    for i in range(start_day, end_day + 1):
        p = [yesterday_rain[0] * x_prob_rain[i - 2] + yesterday_sun[0] * (1 - x_prob_rain[i - 2]),
             yesterday_rain[1] * x_prob_rain[i - 2] + yesterday_sun[1] * (1 - x_prob_rain[i - 2])]
        x_prob_rain.append(p[0])

    return x_prob_rain


# following lines are main function:
evidence_data_add = "data/assign2_umbrella.txt"
start_day = 101
end_day = 150
# the prior distribution on the initial state, P(X0). 50% rainy, and 50% sunny on day 0.
prior = [0.5, 0.5]

x_prob_rain = prediction(evidence_data_add, prior, start_day, end_day)
for i in range(start_day, end_day + 1):
    print("Day " + str(i) + ": rain " + str(x_prob_rain[i - 1]) + ", sunny " + str(1 - x_prob_rain[i - 1]))
