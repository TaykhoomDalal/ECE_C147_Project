import numpy as np

def augment_data(X, y, trim = True, maxpool = True, averaging = True, average = 2, 
                noise = True, sub_sampling = True, sub_sample = 2, reshape = True, separate_noise = True):
    # data augmentation
    X_tot = X
    y_tot = y

    if trim:
        # trim time steps
        X = X[:, :, 0:500]
    
    if separate_noise:
        X_noisy = X + np.random.normal(0.0, 0.5, X.shape)

        X_tot = np.vstack((X_tot, X_noisy))
        y_tot = np.hstack((y_tot, y))

    if maxpool:
        X_tot = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    if averaging:
        X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)

        if noise:
            X_average += np.random.normal(0.0, 0.5, X_average.shape)

        X_tot = np.vstack((X_tot, X_average))
        y_tot = np.hstack((y_tot, y))
    
    if sub_sampling:
        for i in range(sub_sample):
        
            X_subsample = X[:, :, i::sub_sample]
                
            if noise:
                X_subsample += np.random.normal(0.0, 0.5, X_subsample.shape)

            X_tot = np.vstack((X_tot, X_subsample))
            y_tot = np.hstack((y_tot, y))

    if reshape:
        X_tot = np.expand_dims(X_tot, axis = 3)

    return X_tot, y_tot