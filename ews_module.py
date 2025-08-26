import numpy as np 
import tensorflow as tf

from scipy.signal import detrend
from sklearn.metrics.pairwise import pairwise_kernels
from tensorflow.keras.models import load_model

class Stochastic_Resilience():
    def __init__(self, X, type_detrend = 'linear', type_ews = 'variance'):
        self.type_detrend = type_detrend
        self.type_ews = type_ews

        self.X_detrend = detrend(X, axis = 1, type = self.type_detrend)

    def __call__(self):
        if self.type_ews == 'variance':
            if self.X_detrend.shape[0] == 1:
                var = np.var(self.X_detrend.flatten())
            else:
                cov = np.cov(self.X_detrend)
                eigval, eigvec = np.linalg.eig(cov)
                var = np.max(eigval)
            
            return var

        elif self.type_ews == 'lag1-ac':
            # only one-dimensional data
            mean = np.mean(self.X_detrend.flatten())
            X_new = self.X_detrend.flatten() - mean
            ac = np.corrcoef(X_new[:-1], X_new[1:])[0, 1]

            return ac

def MaxEigenvalue_DMD(X, dim_delay = 200, low_rank = 10):
    # delay coordinate
    delayed = []
    for d in range(dim_delay):
        delayed.append(X[:, d:(X.shape[1] - dim_delay + d + 1)])
    X_delay = np.vstack(delayed)

    # DMD
    X1_delay = X_delay[:, :-1]
    X2_delay = X_delay[:, 1:]

    U1, s1, Vh1 = np.linalg.svd(X1_delay, full_matrices = False, compute_uv = True)
    V1 = np.transpose(Vh1)
    if 0 < low_rank < 1:
        cumulative_energy = np.cumsum(s1**2/(s1**2).sum())
        r = np.searchsorted(cumulative_energy, low_rank) + 1
    elif low_rank >= 1 and isinstance(low_rank, int):
        r = min(low_rank, len(s1))
    else:
        r = len(s)
    S1r = np.diag(s1[:r])
    U1r = U1[:, :r]
    V1r = V1[:, :r]

    K = np.transpose(U1r)@X2_delay@V1r@np.linalg.pinv(S1r)
    eigval, eigvec = np.linalg.eig(K)

    return np.max(eigval)

# only one-dimensional data
class EWS_DeepLearning():
    def __init__(self):
        self.model1 = load_model('deep_model/best_model_1_1_len1500.pkl')
        self.model2 = load_model('deep_model/best_model_1_2_len1500.pkl')
        self.model3 = load_model('deep_model/best_model_2_1_len1500.pkl')
        self.model4 = load_model('deep_model/best_model_2_2_len1500.pkl')
        self.model5 = load_model('deep_model/best_model_3_1_len1500.pkl')
        self.model6 = load_model('deep_model/best_model_3_2_len1500.pkl')
        self.model7 = load_model('deep_model/best_model_4_1_len1500.pkl')
        self.model8 = load_model('deep_model/best_model_4_2_len1500.pkl')
        self.model9 = load_model('deep_model/best_model_5_1_len1500.pkl')
        self.model10 = load_model('deep_model/best_model_5_2_len1500.pkl')
        self.model11 = load_model('deep_model/best_model_6_1_len1500.pkl')
        self.model12 = load_model('deep_model/best_model_6_2_len1500.pkl')
        self.model13 = load_model('deep_model/best_model_7_1_len1500.pkl')
        self.model14 = load_model('deep_model/best_model_7_2_len1500.pkl')
        self.model15 = load_model('deep_model/best_model_8_1_len1500.pkl')
        self.model16 = load_model('deep_model/best_model_8_2_len1500.pkl')
        self.model17 = load_model('deep_model/best_model_9_1_len1500.pkl')
        self.model18 = load_model('deep_model/best_model_9_2_len1500.pkl')
        self.model19 = load_model('deep_model/best_model_10_1_len1500.pkl')
        self.model20 = load_model('deep_model/best_model_10_2_len1500.pkl')

    def __call__(self, X, type_detrend = 'linear'):
        X_detrend = detrend(X, axis = 1, type = type_detrend).flatten()
        X_padding = np.zeros(1500 - X_detrend.shape[0])
        X_input = np.concatenate([X_padding, X_detrend])

        sum = 0.0
        cnt = 0.0
        for i in range(X_input.shape[0]):
            sum += np.abs(X_input[i])
            cnt += 1.0
        avg = sum/cnt
        X_input = X_input/avg

        self.X_input = X_input.reshape([1, -1, 1])

    def predict(self):
        probs = self.model1.predict(self.X_input)
        pred = probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model2.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model3.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model4.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model5.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model6.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model7.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model8.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model9.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model10.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model11.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model12.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model13.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model14.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model15.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model16.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model17.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model18.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model19.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]
        probs = self.model20.predict(self.X_input)
        pred += probs[0, 0] + probs[0, 1] + probs[0, 2]

        return pred/20.0

class Koopman_Resilience():
    def __init__(self, X, type_dmd = 'vanilla', dim_delay = 200, low_rank = 0.9, kernel_params = None):
        self.type_dmd = type_dmd
        self.dim_delay = dim_delay
        self.low_rank = low_rank
        self.kernel_params = kernel_params

        X_delay = self._delay_coordinate(X)

        # select observables
        if self.type_dmd == 'vanilla':
            U, s, Vh = np.linalg.svd(X_delay[:, :-1]/(X_delay[:, :-1].shape[1]), full_matrices = False, compute_uv = True)
            V = np.transpose(Vh)
            
            r = self._low_rank_approx(s)
            Sr = np.diag(s[:r])
            Ur = U[:, :r]
            Vr = V[:, :r]

            Psi_X = X_delay[:, :-1]@Vr@np.linalg.pinv(Sr)
            Psi_Y = X_delay[:, 1:]@Vr@np.linalg.pinv(Sr)
        else:
            G_dict = pairwise_kernels(np.transpose(X_delay[:, :-1]), np.transpose(X_delay[:, :-1]), metric = self.type_dmd, **self.kernel_params)
            A_dict = pairwise_kernels(np.transpose(X_delay[:, 1:]), np.transpose(X_delay[:, :-1]), metric = self.type_dmd, **self.kernel_params)
            Sr, Ur = self._svd_gram_matrix(G_dict)

            K_dict = np.linalg.pinv(Sr)@np.transpose(Ur)@A_dict@Ur@np.linalg.pinv(Sr)
            K_eigval, Z = np.linalg.eig(K_dict)
            idx_sorted = np.argsort(-K_eigval.real)
            K_eigval = K_eigval[idx_sorted]
            Z = Z[:, idx_sorted]
            Q, R = np.linalg.qr(Z)

            Psi_X = pairwise_kernels(np.transpose(X_delay[:, :-1]), np.transpose(X_delay[:, :-1]), metric = self.type_dmd, **self.kernel_params)@Ur@np.linalg.pinv(Sr)@Q 
            Psi_Y = pairwise_kernels(np.transpose(X_delay[:, 1:]), np.transpose(X_delay[:, :-1]), metric = self.type_dmd, **self.kernel_params)@Ur@np.linalg.pinv(Sr)@Q 

        # DMD
        self.G = np.transpose(Psi_X)@Psi_X
        self.G = (self.G + np.transpose(self.G))/2.0
        self.A = np.transpose(Psi_X)@Psi_Y
        self.L = np.transpose(Psi_Y)@Psi_Y
        self.L = (self.L + np.transpose(self.L))/2.0
        K_dmd = np.linalg.pinv(self.G)@self.A

        eigval, eigvec = np.linalg.eig(K_dmd)
        idx_sorted = np.argsort(-eigval.real)
        self.eigval = eigval[idx_sorted]
        self.eigvec = eigvec[:, idx_sorted]

    def resDMD(self, eigval, eigvec):
        num = self._hermitian(eigvec)@(self.L - eigval*self._hermitian(self.A) - np.conjugate(eigval)*self.A + eigval*np.conjugate(eigval)*self.G)@eigvec
        den = self._hermitian(eigvec)@self.G@eigvec

        return np.abs(num.real/den.real)

    def resKMD(self):
        reskmd = 0.0
        for i in range(self.eigval.shape[0]):
            reskmd += self.resDMD(self.eigval[i], self.eigvec[:, i])
        
        return reskmd/(self.eigval.shape[0])

    def _hermitian(self, x):
        return np.transpose(np.conjugate(x))

    def _delay_coordinate(self, X):
        delayed = []
        for d in range(self.dim_delay):
            delayed.append(X[:, d:(X.shape[1] - self.dim_delay + d + 1)])
        X_delay = np.vstack(delayed)

        return X_delay

    def _svd_gram_matrix(self, Gram):
        G_eigval, G_eigvec = np.linalg.eig(Gram)

        G_eigval = G_eigval.real
        idx_negative = G_eigval < 0
        G_eigvec[:, idx_negative] *= -1
        G_eigval[idx_negative] *= -1

        s = np.sqrt(G_eigval)
        idx_sorted = np.argsort(-s)
        U_sorted = G_eigvec[:, idx_sorted]
        s_sorted = s[idx_sorted]

        r = self._low_rank_approx(s_sorted)
        U_sorted = U_sorted[:, :r]
        S_sorted = np.diag(s_sorted[:r])

        return S_sorted, U_sorted

    def _low_rank_approx(self, s):
        if 0 < self.low_rank < 1:
            cumulative_energy = np.cumsum(s**2/(s**2).sum())
            return np.searchsorted(cumulative_energy, self.low_rank) + 1
        elif self.low_rank >= 1 and isinstance(self.low_rank, int):
            return min(self.low_rank, len(s))
        else:
            return len(s)