import matplotlib.pyplot as plt


plt.plot([5000, 10000, 20000], [694.1634926019444, 775.9776200948115, 860.529816234852])

plt.ylabel('Agent Score')
plt.xlabel('No of data points')
# plt.legend(['AgentScore'], loc='upper left')
plt.savefig('scorevsdata.png')