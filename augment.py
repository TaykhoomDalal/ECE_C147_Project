import aug
import numpy as np
import preprocessing

def augment_data(data, trim = True, maxpool = True, sub_sample = 2, averaging = True, average = 2, noise = True, sub_sampling = True, reshape = True):
    # data augmentation
    X = data['X_train_valid']
    y = data['y_train_valid']
    Xtest = data['X_test']
    ytest = data['y_test']


    X_tot = data['X_train_valid']
    y_tot = data['y_train_valid']
    X_test_tot = data['X_test']
    y_test_tot = data['y_test']

    if trim:
        # trim time steps
        X = aug.trim_time(X, 0, 500)
        Xtest = aug.trim_time(Xtest, 0, 500)
    
    if maxpool:
        X_tot = np.max(X.reshape(X.shape[0], -1, X.shape[2], sub_sample), axis=3)
        X_test_tot = np.max(Xtest.reshape(Xtest.shape[0], -1, Xtest.shape[2], sub_sample), axis=3)
    
    if averaging:
        X_train_average = np.mean(X.reshape(X.shape[0], -1, X.shape[2], average),axis=3)
        X_test_average = np.mean(Xtest.reshape(Xtest.shape[0], -1, Xtest.shape[2], average),axis=3)

        if noise:
            X_train_average = aug.gaussian_noise(X_train_average, mean = 0.1, std = 0.5)
            X_test_average = aug.gaussian_noise(X_test_average, mean = 0.1, std = 0.5)

        X_tot = np.vstack((X_tot, X_train_average))
        y_tot = np.hstack((y_tot, y))

        X_test_tot = np.vstack((X_test_tot, X_test_average))
        y_test_tot = np.hstack((y_test_tot, ytest))
    
    if sub_sampling:
        for i in range(sub_sample):
        
            X_train_subsample = X[:, i::sub_sample, :] + \
                                (np.random.normal(0.0, 0.5, X[:, i::sub_sample, :].shape) if noise else 0.0)
                
            X_tot = np.vstack((X_tot, X_train_subsample))
            y_tot = np.hstack((y_tot, y))

            X_test_subsample = Xtest[:, i::sub_sample, :] + \
                                (np.random.normal(0.0, 0.5, Xtest[:, i::sub_sample, :].shape) if noise else 0.0)
                
            X_test_tot = np.vstack((X_test_tot, X_test_subsample))
            y_test_tot = np.hstack((y_test_tot, ytest))


    X_tot = np.expand_dims(X_tot, axis = 2)
    X_test_tot = np.expand_dims(X_test_tot, axis = 2)

    return {
            'X_train_valid': X_tot, 'y_train_valid' : y_tot,
            'X_test' : X_test_tot, 'y_test' : y_test_tot
            }

    # # gaussian noise -> trim to 500 -> subsample every 5 
    # data['X_train_valid'] = aug.gaussian_noise(data['X_train_valid'], mean=.1, std=0.5)
    # # data['X_test'] = aug.gaussian_noise(data['X_test'], mean=0.1, std=0.5)

    # data['X_train_valid'] = aug.trim_time(data['X_train_valid'], 0, 500)
    # data['X_test'] = aug.trim_time(data['X_test'], 0, 500)
    
    # data['X_train_valid'] =preprocessing.subsample(data['X_train_valid'], 5)
    # data['X_test'] = preprocessing.subsample(data['X_test'], 5)

    # train_warp = aug.window_warp(data['X_train_valid'])
    # # test_warp = aug.window_warp(data['X_test'])

    # # train_slice = aug.magnitude_warp(data['X_train_valid'])
    # # test_slice = aug.magnitude_warp(data['X_test'])

    # result = np.stack((data['X_train_valid'], train_warp), axis = 1)
    # data['X_train_valid'] = result.reshape(result.shape[0], result.shape[1] * result.shape[2], result.shape[3])

    # # result = np.stack((data['X_test'], test_warp), axis = 1)
    # # data['X_test'] = result.reshape(result.shape[0], result.shape[1] * result.shape[2], result.shape[3])
    # return data