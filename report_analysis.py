import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('results.pkl', 'rb') as f:
    # carica il dizionario dal file
    my_dict = pickle.load(f)


classe = 11
print(my_dict[classe][2])
plt.plot(my_dict[classe][1])
plt.savefig('elleno.png')