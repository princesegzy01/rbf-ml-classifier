import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
# import fcmeans

n_samples = 3000

X = np.concatenate((
    np.random.normal((-2, -2), size=(n_samples, 2)),
    np.random.normal((2, 2), size=(n_samples, 2))
))


fcm = FCM(n_clusters=2)
fcm.fit(X)


fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)


print(fcm_centers)