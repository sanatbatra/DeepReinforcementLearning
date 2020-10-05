from collections import defaultdict
import matplotlib.pyplot as plt


def calculate_optimal_policy(ph, theta):
    value_prev = defaultdict(float)
    value_next = defaultdict(float)
    value_prev[0] = 0.0
    value_prev[100] = 1.0
    value_next[0] = 0.0
    value_next[100] = 1.0
    changed = True
    while changed:
        changed = False
        for state in range(1, 100):
            max_value = 0.0
            for action in range(1, min(100-state, state)+1):
                value = ph*value_prev[state+action] + (1-ph)*value_prev[state-action]
                max_value = max(max_value, value)

            if abs(max_value - value_next[state]) > theta:
                changed = True

            value_next[state] = max_value

        for state in range(1, 100):
            value_prev[state] = value_next[state]

    policy = [0]

    for state in range(1, 100):
        max_value = 0.0
        p = 1
        for action in range(1, min(100 - state, state) + 1):
            value = ph * value_prev[state + action] + (1 - ph) * value_prev[state - action]
            if value > max_value:
                max_value = value
                p = action

        policy.append(p)

    policy.append(0)

    state_list = []
    value_list = []
    for state in range(0, 101):
        state_list.append(state)
        value_list.append(value_next[state])

    plt.plot(state_list, value_list)

    plt.ylabel('Value Estimates')
    plt.xlabel('Capital')
    plt.savefig('ValueEstimates_%s.png' % ph)

    plt.clf()

    plt.plot(state_list, policy)
    plt.ylabel('Final Policy (stake)')
    plt.xlabel('Capital')
    plt.savefig('FinalPolicy_%s.png' % ph)


calculate_optimal_policy(0.25, theta=0.0)
