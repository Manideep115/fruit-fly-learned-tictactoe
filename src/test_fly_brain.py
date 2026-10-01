import numpy as np

from fly_brain import FlyBrain


brain = FlyBrain()

# Random artificial input
input_signal = np.zeros(brain.num_neurons, dtype=np.float32)

# Activate 100 random neurons
random_ids = np.random.choice(
    brain.num_neurons,
    size=100,
    replace=False,
)

input_signal[random_ids] = 1.0

activity, *_ = brain.run(
    input_signal,
    steps=20,
)

print()
print("🪰 Simulation finished")
print("Active neurons:", np.count_nonzero(activity))
print("Total spikes:", int(activity.sum()))